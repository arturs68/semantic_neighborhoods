import argparse
import os
import numpy as np
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_multiple_whitespaces
from tqdm import tqdm
from transformers import LongformerModel, LongformerTokenizer
import jsonlines
import pandas as pd
import torch


def process_text(text):
    def normalize_text(text):
        text = text.encode('ascii', errors='replace').decode('ascii')
        return text
    text = normalize_text(text)
    CUSTOM_FILTERS = [lambda x: x, strip_tags, strip_multiple_whitespaces]
    processed_text = " ".join(list(preprocess_string(text, CUSTOM_FILTERS)))
    return processed_text


def extract_article_list(path):
    all_articles_ids = []
    all_articles_images = []
    all_articles_text = []
    with jsonlines.open(path) as reader:
        for obj in reader:
            all_articles_ids.append(obj["id"])
            all_articles_images.append(obj["images"])
            all_articles_text.append(obj["text"])
    return all_articles_text, all_articles_ids, all_articles_images


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def main(dataset_directory, jsonlines_filename):
    dataset, ids, images = extract_article_list(os.path.join(dataset_directory, jsonlines_filename))
    print(f'Len dataset = {len(dataset)}')

    text_model = LongformerModel.from_pretrained("allenai/longformer-base-4096").to("cuda")
    text_model.eval()
    tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")

    # pool = Pool(processes=48)
    # processed_text = list(tqdm(pool.map(process_text, dataset), total=len(dataset)))
    # pool.close()
    batch_size=8
    all_embeddings_avg = np.zeros((len(dataset), 768), dtype=np.float)
    for i, chunk in tqdm(enumerate(chunks(dataset, batch_size)), total=len(dataset)/batch_size):
        with torch.no_grad():
            tokenized_text = tokenizer(chunk, return_tensors="pt", truncation=True, padding="max_length")
            model_out = text_model(**(tokenized_text.to("cuda")))
            all_embeddings_avg[i*batch_size: i*batch_size + len(chunk), :] = torch.mean(model_out[0], dim=1).cpu().numpy()

    data_df = pd.DataFrame(zip(ids, images, all_embeddings_avg))
    data_df.to_pickle(os.path.join(dataset_directory, f"longformer_{jsonlines_filename.split('.')[0]}.pkl"))


if __name__ == '__main__':
    # assumes that the dataset directory is
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_directory", type=str)
    parser.add_argument("--jsonlines_filename", type=str)
    args = parser.parse_args()
    main(args.dataset_directory, args.jsonlines_filename)
