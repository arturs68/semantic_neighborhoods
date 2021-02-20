import numpy as np
import os
import sys
import random
import glob
import torch
from PIL import Image
from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter
import csv, itertools
import pandas as pd

csv.field_size_limit(sys.maxsize)

image_features_model = "resnet152"

datasets_dict = {
    "breakingnews": {"data_dir": "/mnt/localdata/szalata/datasets/breakingnews",
                     "jsonl_prefix": "breaking_news"},
    "mscoco": {"data_dir": "/mnt/localdata/szalata/datasets/mscoco",
               "jsonl_prefix": "coco"},
    # not extracted yet
    "politics": {},
    "nytimes": {},
    "goodnews": {}
}

splits = ["train", "val", "test"]


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
        self.article_vectors = doc2vec_df[2]
        doc2vec_df = doc2vec_df.reset_index()[["index", 1]].explode(1)
        doc2vec_df["index"] = doc2vec_df["index"].astype("category")
        doc2vec_df[1] = doc2vec_df[1].astype("category")
        self.id_path = doc2vec_df
        self.image_vectors = pd.read_pickle(os.path.join(dataset_dict["data_dir"],
                                                         f"{image_features_model}_{dataset_dict['jsonl_prefix']}_{split}.pkl"))

    def __len__(self):
        return len(self.id_path)

    def __getitem__(self, index):
        entry = self.id_path.iloc[index]
        return self.image_vectors[entry[1]].astype(np.float32), self.article_vectors.iloc[entry["index"]].astype(np.float32)


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
    train_dataloader = torch.utils.data.DataLoader(dataset=MyDataset('train', "breakingnews"),
                                                   batch_size=512, shuffle=True, num_workers=40)
    test_dataloader = torch.utils.data.DataLoader(dataset=MyDataset('val', "breakingnews"),
                                                  batch_size=1024, shuffle=False, num_workers=40)
    # writer = SummaryWriter(f'models/')
    img_model = torch.nn.DataParallel(ImageProjectModel()).cuda()
    text_model = torch.nn.DataParallel(TextProjectModel()).cuda()
    optimizer = torch.optim.Adam(
        params=itertools.chain(img_model.parameters(), text_model.parameters()), lr=0.0001,
        weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', verbose=True,
                                                           patience=5)
    itr = 0
    triplet_loss = TripletLoss()
    best_loss = sys.maxsize
    for e in tqdm(range(1, 1000), ascii=True, desc='Epoch'):
        img_model.train()
        text_model.train()
        with tqdm(total=len(train_dataloader), ascii=True, leave=False, desc='iter') as pbar:
            for i, (images, articles) in enumerate(train_dataloader):
                itr += 1
                optimizer.zero_grad()
                image_projections = img_model(images.cuda())
                article_projections = text_model(articles.cuda())
                loss = triplet_loss(image_projections, article_projections)

                loss.backward()
                optimizer.step()
                # writer.add_scalar('data/train_loss', loss.item(), itr)
                # writer.add_scalar('data/l_i2t', l_i2t.item(), itr)

                pbar.update()
        img_model.eval()
        text_model.eval()
        losses = []
        random.seed(9485629)
        with tqdm(total=len(test_dataloader), ascii=True, leave=False,
                  desc='eval') as pbar, torch.no_grad():
            for i, (images, articles) in enumerate(test_dataloader):
                with torch.no_grad():
                    image_projections = img_model(images.cuda())
                    article_projections = text_model(articles.cuda())
                    loss = triplet_loss(image_projections, article_projections)

                # writer.add_scalar('data/train_loss', loss.item(), itr)
                # writer.add_scalar('data/l_i2t', l_i2t.item(), itr)

                pbar.update()
                losses.append(loss.item())

                pbar.update()
        curr_loss = np.mean(losses)
        # writer.add_scalar('data/val_loss', curr_loss, e)
        scheduler.step(curr_loss)
        # save only the best model
        if curr_loss < best_loss:
            best_loss = curr_loss
            print(best_loss)
        #     # delete prior models
        #     prior_models = glob.glob(f'models/l_i2t_l_sym_{sys.argv[1].replace(".","_")}_l_img_{sys.argv[2].replace(".","_")}_l_text_{sys.argv[3].replace(".","_")}/*.pth')
        #     for pm in prior_models:
        #         os.remove(pm)
        #     try:
        #         torch.save(rnn_model.state_dict(), f'models/l_i2t_l_sym_{sys.argv[1].replace(".","_")}_l_img_{sys.argv[2].replace(".","_")}_l_text_{sys.argv[3].replace(".","_")}/rnn_model_{e}.pth')
        #         torch.save(img_model.state_dict(), f'models/l_i2t_l_sym_{sys.argv[1].replace(".","_")}_l_img_{sys.argv[2].replace(".","_")}_l_text_{sys.argv[3].replace(".","_")}/img_model_{e}.pth')
        #     except:
        #         print('Failed saving')
        #         continue


if __name__ == '__main__':
    main()
