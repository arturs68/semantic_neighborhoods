import argparse
import os, random


import jsonlines
from multiprocessing import Pool
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_multiple_whitespaces, remove_stopwords, strip_short, strip_non_alphanum

# Set up log to terminal
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)
from tqdm import tqdm


def convert_text(text):
    def normalize_text(text):
        text = text.encode('ascii', errors='replace').decode('ascii')
        return text
    text = normalize_text(text)
    CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_non_alphanum, remove_stopwords, strip_short]
    tokenized_text = list(preprocess_string(text, CUSTOM_FILTERS))
    return tokenized_text


def extract_article_list(path):
    all_articles = []
    with jsonlines.open(path) as reader:
        for obj in reader:
            all_articles.append(obj["text"])
    return all_articles


def main(dataset_directory, jsonlines_filename):
    pool = Pool(processes=48)
    all_train_text = extract_article_list(os.path.join(dataset_directory, jsonlines_filename))
    documents = list(tqdm(pool.imap(convert_text, all_train_text, chunksize=1), total=len(all_train_text)))
    pool.close()
    random.shuffle(documents)
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(documents)]
    print('Starting training on {} documents'.format(len(documents)))
    d2v = Doc2Vec(
        documents=documents, vector_size=512, workers=48, epochs=50, window=20, min_count=20)
    d2v.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    d2v.save(os.path.join(
        dataset_directory, "doc2vec",
        f'doc2vec_model_{dataset_directory.split("/datasets/")[1]}.gensim'))
    print('Finished training')


if __name__ == '__main__':
    # assumes that the dataset directory is
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_directory", type=str)
    parser.add_argument("--jsonlines_filename", type=str)
    args = parser.parse_args()
    main(args.dataset_directory, args.jsonlines_filename)
