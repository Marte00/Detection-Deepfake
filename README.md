# Detection-Deepfake

## I. Introduction

La détection d’images Deepfake est une tâche cruciale dans la lutte contre la désinformation visuelle. Ce projet vise à développer une solution permettant de détecter automatiquement si une image a été générée par une intelligence artificielle, en l’occurrence un modèle de Stable Diffusion. 

## II. DATA et Pré-traitement
Les données utilisées pour l'entraînement proviennent du dataset [https://huggingface.co/datasets/ChristophSchuhmann/improved_aesthetics_6plus](url) pour les images réelles et [https://huggingface.co/datasets/poloclub/diffusiondb](url) pour les images fakes.
notre dataset est composé de 2000 images en total équilibré sur 2 classes de type réelles (0) ou fakes (1). Il sera ensuite séparé en ensemble de training, validation et de test dans les proportions 80%/10%/10% tous équilibrés.

Afin de tester si notre modèle généralise sur des images générées par d'autres modèles inconnus, nous introduisons 3 ensembles (200 images par ensembles) de tests contenant respectivement des images de Midjourney, Dalle-E et Stable Diffusion V1.5. Les images réelles sont prises au hasard dans ImageNet.

### a) Augmentation des donnés
Un Horizontal Flip, Vertical Flip et découpage d'une image en 4 sous images (patch) sont les principales techniques utilisées pour l'augmentation des données.

### b) Pré-traitements
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
## III. Finetuning avec ResNet50
Le modèle est chargé avec les poids entrainés sur Imagenet1K, une modification de la couches Fully Connected est apporté afin que la sorie soit de 2 classes au lieu de 1000 classes.

```ruby
#chargement du modèle avec les poids pré-entrainés.
model = models.resnet50(weights = models.ResNet50_Weights.DEFAULT).to(device)

# fige les poids du modèle 
for param in model.parameters():
 param.requires_grad = False

# modification de la couche FC
model.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, 2)).to(device)

# On fige ou pas un ou plusieurs couches de ResNet50
layer1 = "layer4"
layer2 = "layer3"
#layer3 = "layer2"
for name, param in model.named_parameters():
    if layer1 in name:
      param.requires_grad = True
    if layer2 in name:
      param.requires_grad = True
    #if layer3 in name:
      #param.requires_grad = True
    print(name,param.requires_grad)
```
La fonction de perte utilisée est la CrossEntropyLoss avec l'optimiseur Adam, le learning rate est à 0.001 : 

```ruby
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)
```

Entrainement du modèle : 
```ruby
model_trained, val_loss, train_loss = train_model(model, criterion, optimizer, num_epochs=20)
model = model_trained
```
## IV. Evauation & Expérimentation

Pour l'évaluatiob du modèle les métriques classiques précision, rappel et F1-score sont utilisés.

```ruby
from sklearn.metrics import precision_recall_fscore_support

model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1)
        _, predicted = torch.max(probs, 1)

        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')

print(f'Precision: {precision:.6f}')
print(f'Recall: {recall:.6f}')
print(f'F1 Score: {f1_score:.6f}')
```
![alt text](https://github.com/Marte00/Detection-Deepfake/image_résultat/image1.png?raw=true)


## V. Conclusion
