import argparse
import os
from tqdm import tqdm
from transformers import LongformerTokenizer
import jsonlines
import pandas as pd
import torch
import numpy as np

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

    tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")

    batch_size=512
    all_tokens = np.zeros((len(dataset), 2, 4096), dtype=np.float)
    for i, chunk in tqdm(enumerate(chunks(dataset, batch_size)), total=len(dataset)/batch_size):
        with torch.no_grad():
            tokenized_text = tokenizer(chunk, return_tensors="pt", truncation=True, padding="max_length")
            all_tokens[i*batch_size: i*batch_size + len(chunk), 0, :] = tokenized_text["input_ids"].numpy()
            all_tokens[i*batch_size: i*batch_size + len(chunk), 1, :] = tokenized_text["attention_mask"].numpy()

    data_df = pd.DataFrame(zip(ids, images, all_tokens.astype(np.int_)))
    data_df.to_pickle(os.path.join(dataset_directory, f"longformer_tokens_{jsonlines_filename.split('.')[0]}.pkl"))


if __name__ == '__main__':
    # assumes that the dataset directory is
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_directory", type=str)
    parser.add_argument("--jsonlines_filename", type=str)
    args = parser.parse_args()
    main(args.dataset_directory, args.jsonlines_filename)
