# Detection-Deepfake

## Introduction

La détection d’images Deepfake est une tâche cruciale dans la lutte contre la désinformation visuelle. Ce projet vise à développer une solution permettant de détecter automatiquement si une image a été générée par une intelligence artificielle, en l’occurrence un modèle de Stable Diffusion. 

## DATA et Pré-traitement
Les données utilisées pour l'entraînement proviennent du dataset [https://huggingface.co/datasets/ChristophSchuhmann/improved_aesthetics_6plus](url) pour les images réelles et [https://huggingface.co/datasets/poloclub/diffusiondb](url) pour les images fakes.
notre dataset est composé de 2000 images en total équilibré sur 2 classes de type réelles (0) ou fakes (1). Il sera ensuite séparé en ensemble de training, validation et de test dans les proportions 80%/10%/10% tous équilibrés.

### Augmentation des donnés
Un Horizontal Flip, Vertical Flip et découpage d'une image en 4 sous images (patch) sont les principales techniques utilisées pour l'augmentation des données.

### Pré-traitements
Les images sont redimentionnées, normalisé, application du RandomAffine et par la suite un Transformé de Fourier.

A noté: le dataset [https://huggingface.co/datasets/ChristophSchuhmann/improved_aesthetics_6plus](url) n'est plus disponible malheureusement.

```ruby
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.fft import fft2

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

data_transforms = {
    'train':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        normalize
    ]),
    'validation':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
    ])
}


class FourierTransform(object):
    def __call__(self, tensor):
        dft = fft2(tensor)
        return dft.real

    def __repr__(self):
        return self.__class__.__name__ + '()'

data_transforms2 = {
    'train': transforms.Compose([
        #FourierTransform(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        FourierTransform()
    ]),
    'validation': transforms.Compose([
        #FourierTransform(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        FourierTransform()
    ])
}
```
## Finetuning avec ResNet50

## Evauation & Expérimentation

## Conclusion
