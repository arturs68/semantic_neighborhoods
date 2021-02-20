import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from tqdm import tqdm

image_batch_size = 500

class Img2Vec():
  RESNET_OUTPUT_SIZES = {
    'resnet18': 512,
    'resnet34': 512,
    'resnet50': 2048,
    'resnet101': 2048,
    'resnet152': 2048,
  }

  def __init__(self, cuda=True, model='resnet-152', layer='default', layer_output_size=2048):
    """ Img2Vec
    :param cuda: If set to True, will run forward pass on GPU
    :param model: String name of requested model
    :param layer: String or Int depending on model.  See more docs: https://github.com/christiansafka/img2vec.git
    :param layer_output_size: Int depicting the output size of the requested layer
    """
    self.device = torch.device("cuda" if cuda else "cpu")
    self.layer_output_size = layer_output_size
    self.model_name = model

    self.model, self.extraction_layer = self._get_model_and_layer(model, layer)

    self.model = self.model.to(self.device)

    self.model.eval()

    self.scaler = transforms.Resize((224, 224))
    self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])
    self.to_tensor = transforms.ToTensor()

  def chunks(self, lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
      yield lst[i:i + n]

  def get_vec(self, img, tensor=True):
    """ Get vector embedding from PIL image
    :param img: PIL Image or list of PIL Images
    :param tensor: If True, get_vec will return a FloatTensor instead of Numpy array
    :returns: Numpy ndarray
    """
    if type(img) == list:
      all_embeddings = np.zeros((len(img), 2048), dtype=np.float)
      for i, chunk in tqdm(enumerate(self.chunks(img, image_batch_size)), total=len(img)/image_batch_size):
        a_chunk = []
        for im in chunk:
          try:
            a_chunk.append(self.normalize(self.to_tensor(self.scaler(Image.open(im)))))
          except:
            print(f"{im} is grayscale. converting to rgb")
            a_chunk.append(self.normalize(self.to_tensor(self.scaler(Image.open(im).convert('RGB')))))
        images = torch.stack(a_chunk).to(self.device)
        my_embedding = torch.zeros(len(a_chunk), self.layer_output_size, 1, 1)

        def copy_data(m, i, o):
          my_embedding.copy_(o.data)

        h = self.extraction_layer.register_forward_hook(copy_data)
        with torch.no_grad():
          h_x = self.model(images)
        h.remove()

        if tensor:
          return my_embedding
        else:
          all_embeddings[i*image_batch_size: i*image_batch_size + len(a_chunk), :] = my_embedding.numpy()[:, :, 0, 0]
      return all_embeddings
    else:
      image = self.normalize(self.to_tensor(self.scaler(Image.open(img)))).unsqueeze(0).to(self.device)

      if self.model_name == 'alexnet':
        my_embedding = torch.zeros(1, self.layer_output_size)
      else:
        my_embedding = torch.zeros(1, self.layer_output_size, 1, 1)

      def copy_data(m, i, o):
        my_embedding.copy_(o.data)

      h = self.extraction_layer.register_forward_hook(copy_data)
      h_x = self.model(image)
      h.remove()

      if tensor:
        return my_embedding
      else:
        if self.model_name == 'alexnet':
          return my_embedding.numpy()[0, :]
        else:
          return my_embedding.numpy()[0, :, 0, 0]

  def _get_model_and_layer(self, model_name, layer):
    """ Internal method for getting layer from model
    :param model_name: model name such as 'resnet-18'
    :param layer: layer as a string for resnet-18 or int for alexnet
    :returns: pytorch model, selected layer
    """

    if model_name == 'resnet-152':
      model = models.resnet152(pretrained=True)
      model.eval()
      if layer == 'default':
        layer = model._modules.get('avgpool')
        self.layer_output_size = 2048
      else:
        layer = model._modules.get(layer)

      return model, layer

    else:
      raise KeyError('Model %s was not found' % model_name)