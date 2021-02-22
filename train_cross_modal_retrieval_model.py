import argparse

import numpy as np
import os
import sys
import torch
from tqdm import tqdm
import csv, itertools
import pandas as pd
import mlflow

csv.field_size_limit(sys.maxsize)

image_features_model = "resnet152"

datasets_dict = {
    "breakingnews": {"data_dir": "/mnt/localdata/szalata/datasets/breakingnews",
                     "jsonl_prefix": "breaking_news"},
    "mscoco": {"data_dir": "/mnt/localdata/szalata/datasets/mscoco",
               "jsonl_prefix": "coco"},
    "nytimes": {"data_dir": "/mnt/localdata/szalata/datasets/nytimes",
                "jsonl_prefix": "nytimes800k"},
    # not extracted yet
    "politics": {},
    "goodnews": {}
}

splits = ["train", "val", "test"]


def compute_metrics(ranks):
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    return (r1, r5, r10, medr, meanr)


def t2i(images, captions):
    """
    Text->Images (Image Search)
    """

    ranks = np.zeros(len(images))
    for index in tqdm(range(len(captions))):
        query = captions[index]
        # Compute scores
        ranks_all = np.argsort(np.dot(query, images.T))[::-1]
        ranks[index] = np.where(ranks_all == index)[0][0]

    return compute_metrics(ranks)


def compute_embeddings(img_model, text_model, dataloader, batch_size):
    img_embeddings = np.zeros((len(dataloader.dataset), 256), dtype=np.float)
    text_embeddings = np.zeros((len(dataloader.dataset), 256), dtype=np.float)
    img_model.eval()
    text_model.eval()

    with tqdm(total=len(dataloader), ascii=True, leave=False, desc='iter') as pbar:
        for i, (images, articles) in enumerate(dataloader):
            with torch.no_grad():
                image_projections = img_model(images.float().cuda())
                article_projections = text_model(articles.float().cuda())

            img_embeddings[i * batch_size: i * batch_size + len(image_projections)] = image_projections.cpu().numpy()
            text_embeddings[i * batch_size: i * batch_size + len(article_projections)] = article_projections.cpu().numpy()
            pbar.update()

    return img_embeddings, text_embeddings


def set_seed(seed):
    """
    Set the seed for NumPy and pyTorch
    :param args: Namespace
        Arguments passed to the script
    """
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, split, dataset_name):
        super().__init__()
        if dataset_name not in datasets_dict:
            raise Exception(
                f"{dataset_name} not in dataset_dict. \nChoose one of {datasets_dict.keys()}.")
        if split not in splits:
            raise Exception(
                f"{split} not correct. \nChoose one of {splits}.")
        dataset_dict = datasets_dict[dataset_name]
        doc2vec_df = pd.read_pickle(os.path.join(dataset_dict["data_dir"], "doc2vec",
                                                 f"doc2vec_{dataset_dict['jsonl_prefix']}_{split}.pkl"))
        self.article_vectors = np.array(doc2vec_df[2].tolist())
        doc2vec_df["index"] = list(range(len(doc2vec_df)))
        doc2vec_df = doc2vec_df[["index", 1]].explode(1)
        doc2vec_df["index"] = doc2vec_df["index"].astype("category")
        doc2vec_df[1] = doc2vec_df[1].astype("category")
        image_df = pd.DataFrame(pd.read_pickle(os.path.join(dataset_dict["data_dir"],
                                                            f"{image_features_model}_{dataset_dict['jsonl_prefix']}_{split}.pkl"))).T
        self.id_df = pd.merge(doc2vec_df.rename(columns={1: "path"}),
                              image_df.reset_index().reset_index()[["level_0", "index"]],
                              left_on="path", right_on="index").drop(columns=["index_y"])
        self.image_vectors = image_df.values

    def __len__(self):
        return len(self.id_df)

    def __getitem__(self, index):
        entry = self.id_df.iloc[index]
        return self.image_vectors[entry["level_0"]], self.article_vectors[entry["index_x"]]


class ImageProjectModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = torch.nn.SELU()
        self.dropout = torch.nn.Dropout(p=0.1)
        self.projector_1 = torch.nn.Linear(2048, 1024)
        self.projector_2 = torch.nn.Linear(1024, 256)

    def forward(self, input):
        out = self.projector_2(self.dropout(self.activation(self.projector_1(input))))
        return out


class TextProjectModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = torch.nn.SELU()
        self.dropout = torch.nn.Dropout(p=0.1)
        self.projector_1 = torch.nn.Linear(512, 512)
        self.projector_2 = torch.nn.Linear(512, 256)

    def forward(self, input):
        out = self.projector_2(self.dropout(self.activation(self.projector_1(input))))
        return out


def cosine_sim(im, s):
    """
  Cosine similarity between all the image and sentence pairs
  """
    return im.mm(s.t())


class TripletLoss(torch.nn.Module):
    """
  Compute contrastive loss
  """

    def __init__(self, margin=1, max_violation=False, i2t=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.i2t = i2t
        self.sim = cosine_sim
        self.max_violation = max_violation

    def forward(self, im, s):
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        I = torch.eye(scores.size(0)) > .5
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        loss = cost_im.sum()
        if self.i2t:
            loss += cost_s.sum()

        return loss


def main():
    parser = argparse.ArgumentParser()

    # Required parameter
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where the model predictions and checkpoints "
                             "will be written.")

    parser.add_argument("--train", action='store_true', help="Whether to run training.")
    parser.add_argument("--eval", action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=1e-5, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_epochs", default=100, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size per GPU/CPU.")
    parser.add_argument("--batch_size_eval", default=256, type=int, help="Batch size per GPU/CPU.")
    parser.add_argument("--run_name", type=str, help="name of the mlflow run")
    parser.add_argument("--overwrite_output_dir", action='store_true',
                        help="Overwrite the content of the output directory")

    args = parser.parse_args()
    mlflow.set_experiment("article2image")
    mlflow.start_run(run_name=args.run_name)
    mlflow.log_params(vars(args))

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and not args.overwrite_output_dir and args.train:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to "
            "overcome.".format(
                args.output_dir))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    best_model_img = os.path.join(args.output_dir, "best_model_img.model")
    best_model_text = os.path.join(args.output_dir, "best_model_text.model")

    set_seed(args.seed)
    n_gpu = torch.cuda.device_count()
    batch_size = args.batch_size * max(1, n_gpu)
    batch_size_eval = args.batch_size * max(1, n_gpu)

    test_dataloader = torch.utils.data.DataLoader(dataset=MyDataset('val', args.dataset_name),
                                                  batch_size=batch_size_eval, shuffle=False,
                                                  num_workers=4)
    img_model = torch.nn.DataParallel(ImageProjectModel()).cuda()
    text_model = torch.nn.DataParallel(TextProjectModel()).cuda()
    if args.train:
        train_dataloader = torch.utils.data.DataLoader(
            dataset=MyDataset('train', args.dataset_name),
            batch_size=batch_size, shuffle=True, num_workers=32)

        optimizer = torch.optim.Adam(
            params=itertools.chain(img_model.parameters(), text_model.parameters()),
            lr=args.learning_rate,
            weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', verbose=True,
                                                               patience=5)
        itr = 0
        triplet_loss = TripletLoss()
        best_loss = sys.maxsize
        for e in tqdm(range(1, args.num_epochs + 1), ascii=True, desc='Epoch'):
            img_model.train()
            text_model.train()
            with tqdm(total=len(train_dataloader), ascii=True, leave=False, desc='iter') as pbar:
                for i, (images, articles) in enumerate(train_dataloader):
                    itr += 1
                    optimizer.zero_grad()
                    image_projections = img_model(images.float().cuda())
                    article_projections = text_model(articles.float().cuda())
                    loss = triplet_loss(image_projections, article_projections)

                    loss.backward()
                    optimizer.step()
                    if itr % 100 == 0:
                        mlflow.log_metric("training loss", loss.item() / max((len(
                            image_projections) * len(article_projections) - len(image_projections)), 1),
                                          itr)

                    pbar.update()
            img_model.eval()
            text_model.eval()
            losses = []
            with tqdm(total=len(test_dataloader), ascii=True, leave=False,
                      desc='eval') as pbar, torch.no_grad():
                for i, (images, articles) in enumerate(test_dataloader):
                    with torch.no_grad():
                        image_projections = img_model(images.float().cuda())
                        article_projections = text_model(articles.float().cuda())
                        loss = triplet_loss(image_projections, article_projections)

                    pbar.update()
                    losses.append(loss.item() / max(
                        (len(image_projections) * len(article_projections) - len(image_projections)),
                        1))

                    pbar.update()
            test_loss = np.mean(losses)
            mlflow.log_metric("test loss", test_loss, e)
            scheduler.step(test_loss)
            # save only the best model
            if test_loss < best_loss:
                best_loss = test_loss
                if os.path.exists(best_model_img):
                    os.remove(best_model_img)
                if os.path.exists(best_model_text):
                    os.remove(best_model_text)
                torch.save(img_model.state_dict(), best_model_img)
                torch.save(text_model.state_dict(), best_model_text)

    if args.eval:
        img_model.load_state_dict(torch.load(best_model_img))
        text_model.load_state_dict(torch.load(best_model_text))
        embeddings_img, embeddings_cap = compute_embeddings(img_model, text_model, test_dataloader, batch_size_eval)
        recall = t2i(embeddings_img, embeddings_cap)
        avg_recall = (recall[0] + recall[1] + recall[2]) / 3
        print("Average t2i Recall: %.1f" % avg_recall)
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" % recall)


if __name__ == '__main__':
    main()
