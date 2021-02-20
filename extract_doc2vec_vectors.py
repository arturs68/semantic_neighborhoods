import argparse
import os
from multiprocessing import Pool
from gensim.models.doc2vec import Doc2Vec
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_multiple_whitespaces, remove_stopwords, strip_short, strip_non_alphanum
from tqdm import tqdm
import jsonlines
import pandas as pd


def process_text(tup):
    def normalize_text(text):
        text = text.encode('ascii', errors='replace').decode('ascii')
        return text
    id, text, imgs = tup
    text = normalize_text(text)
    CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_non_alphanum, remove_stopwords, strip_short]
    tokenized_text = list(preprocess_string(text, CUSTOM_FILTERS))
    if tokenized_text:
        global d2v
        features = d2v.infer_vector(tokenized_text)
        return id, imgs, features
    else:
        return id, imgs, None


def extract_article_list(path):
    all_articles = []
    with jsonlines.open(path) as reader:
        for obj in reader:
            all_articles.append((obj["id"], obj["text"], obj["images"]))
    return all_articles


def main(dataset_directory, jsonlines_filename):
    dataset = extract_article_list(os.path.join(dataset_directory, jsonlines_filename))
    print(f'Len dataset = {len(dataset)}')
    global d2v
    d2v = Doc2Vec.load(os.path.join(
        dataset_directory, "doc2vec",
        f'doc2vec_model_{dataset_directory.split("/datasets/")[1]}.gensim'))
    pool = Pool(processes=48)
    pth_to_features = list(tqdm(pool.imap_unordered(process_text, dataset, chunksize=1), total=len(dataset)))
    pool.close()
    data_df = pd.DataFrame([(article_id, imgs, features.ravel()) for article_id, imgs, features in pth_to_features if features is not None])
    data_df.to_pickle(os.path.join(dataset_directory, "doc2vec", f"doc2vec_{jsonlines_filename.split('.')[0]}.pkl"))


if __name__ == '__main__':
    # assumes that the dataset directory is
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_directory", type=str)
    parser.add_argument("--jsonlines_filename", type=str)
    args = parser.parse_args()
    main(args.dataset_directory, args.jsonlines_filename)
