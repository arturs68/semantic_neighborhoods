import argparse
import os, pickle
from multiprocessing import Pool
from gensim.models.doc2vec import Doc2Vec
from tqdm import tqdm
import jsonlines
import pandas as pd
from img2vec import Img2Vec


def extract_image_list(path):
  all_images = []
  with jsonlines.open(path) as reader:
    for obj in reader:
      for img_path in obj["images"]:
        all_images.append(img_path)
  return all_images


def main(dataset_directory, jsonlines_filename):
  img_paths = list(set(extract_image_list(os.path.join(dataset_directory, jsonlines_filename))))
  print(f'Len dataset = {len(img_paths)}')
  img2vec = Img2Vec()
  if dataset_directory.split("/")[-1] == "mscoco":
    features = img2vec.get_vec(list(map(lambda x: os.path.join("/mnt/storage01/datasets/coco/images", x), img_paths)), tensor=False)
  else:
    features = img2vec.get_vec(list(map(lambda x: os.path.join(dataset_directory, x), img_paths)), tensor=False)
  pth_to_features = dict([(pth, features[i]) for i, pth in enumerate(img_paths)])
  with open(os.path.join(dataset_directory, f"resnet152_{jsonlines_filename.split('.')[0]}.pkl"), 'wb') as handle:
    pickle.dump(pth_to_features, handle)

if __name__ == '__main__':
  # assumes that the dataset directory is
  parser = argparse.ArgumentParser()

  parser.add_argument("--dataset_directory", type=str)
  parser.add_argument("--jsonlines_filename", type=str)
  args = parser.parse_args()
  main(args.dataset_directory, args.jsonlines_filename)