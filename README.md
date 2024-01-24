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

Les résulats montre que ResNet50 a une meilleur performance comparé à EfficientNetV2, le pré-entrainainement améliore le score du modèle.

![Screenshot_2](https://github.com/Marte00/Detection-Deepfake/assets/107618271/732f299e-8c4e-41ae-b32d-899caed6c2d9)

Nous avons comparé les résulats en fonction de la partie du réseau qui est entraîné voici les résulats que nous avons obtenus: 
![image2](https://github.com/Marte00/Detection-Deepfake/assets/107618271/af666a93-839d-420c-acd8-1047b02711e8)


Nous pouvons constater que la courbe de loss ne converge pas bien et que le modèle n'apprend pas sur notre dataset, cependant on remarque de bon résultat qui sont dû au pré-entraînement du modèle ResNet sur ImageNet. Cela peut être dû à notre dataset qui est petit et que les poids de ImageNet prennent le dessus.

![image3](https://github.com/Marte00/Detection-Deepfake/assets/107618271/6ad1088b-cfe7-4c76-bee5-fa093ff3b0e1)
![image4](https://github.com/Marte00/Detection-Deepfake/assets/107618271/1fd4e97c-7bf1-4690-b8b7-748f2dbdbaf2)
![image5](https://github.com/Marte00/Detection-Deepfake/assets/107618271/ee2d4d9f-ed45-4fdc-8085-3ccc568dac7e)


Afin de visualiser la zone que le modèle regarde nous avons généré le heat map de chaque couche du réseau, nous pouvons observer que sur la couche d'entrée le modèle se concentre sur l'ensemble de l'image au fur et à mesure une zone précise se dessine et montre la partie qui est prise en compte par le modèle. 
![image6](https://github.com/Marte00/Detection-Deepfake/assets/107618271/2ddea18c-1cc9-450c-98f1-18f5a3d446b9)

Sur les ensembles inconnus, nous voyons que notre modèle ne généralise pas car le modèle se trompe beaucoup sur Midjourney et Dalle-E bien qu' avec stable diffusion nous avons 23 sur 100 d'images bien classées le modèle a du mal. Cependant pour les images réelles, elles sont toutes bien classées car le modèle est pré-entraîné sur ImageNet. Sans augmentation des données, on voit une nette amélioration sur les prédictions bien qu'il reste un grand nombre d'erreurs il y a une amélioration. L'apprentissage avec transformé de Fourier améliore la prédiction sur stable diffusion cependant les résultats sont mitigés sur les ensembles Dalle-E et Midjourney, de plus on perd en nombre d'image réelle bien classée qui peut être dû que les images réelles sont variés et complexe donc difficile d'être caractérisé seulement par le transformé de Fourier. Il faudrait diversifier et augmenter le dataset pour tester et voir l'impact afin de comparer et déterminer ce qui marche ou ne marche pas.

Modèle entraîné avec augmentation des données:
![image7](https://github.com/Marte00/Detection-Deepfake/assets/107618271/a5d60f17-a7d8-47d0-ba4d-8855a778543a)

Modèle entraîné sans augmentation des données:
![image8](https://github.com/Marte00/Detection-Deepfake/assets/107618271/733592bd-df78-4cb5-a0d6-42c86d0b4472)

Modèle entraîné sur le Transformé de Fourier des images:
![image10](https://github.com/Marte00/Detection-Deepfake/assets/107618271/0b6d9154-7013-4205-a560-e304b8a5e5c6)

## V. Conclusion

Nous avons pu mettre en avant l'importance sur le choix des couches d’un réseau à entraîné, le finetuning sur un modèle pré-entraîné permet d'améliorer les scores et traiter les images en coupant l’image afin d’apprendre sur une zone locale améliore les résultats d'après les résultats de l'entraînement enfin le transformé de fourier peut apporter un indice supplémentaire dans la détection de fake par l'étude du domaine spatial d'une image. Cependant nous avons pu constater les limites de notre méthode, notamment une mauvaise généralisation du modèle et l'apprentissage sur le dataset, ce point pourrait être amélioré et développé afin de comparer les résultats sur un plus grand dataset. Le modèle ResNet50 est entraîné sur des images de tailles 224x224, nous avons dû réduire la taille de nos images cependant le redimensionnement des images en entrées peut effacer les traces subtiles de haute fréquence laissées par le processus de génération des modèles, donc nous pouvons perdre des informations importantes. De plus, il faut prendre en compte que les images sont diverses et que notre méthode qui se concentre sur l'analyse générale d'une image peut être compliqué face à des images de haute résolutions mais aussi des images d'oeuvres d'arts qui sont abstraites rend les choses difficiles dans la reconnaissance, de plus la majorité des images réelles sont souvent modifier par un utilisateur à travers des retouches, filtres etc. Mais aussi la compression JPEG, PNG et autres qui peuvent apporter des informations plus ou moins utiles. Il est intéressant de se pencher sur une manière de catégoriser nos images afin de peut-être entraîner un modèle spécialisé. Une autre idée serait peut être d'apporter plus d'information en introduisant les prompts qui apportent l'information textuelle et description d'une image. Ou se tourner vers une approche auto-supervisé par le calcul de le l’erreur de reconstruction de diffusion **DIRE**, potentiellement développer une tâche prétexte afin d’apprendre la manière dont les modèles de diffusion génère les images. Basé sur l’article **DIRE for Diffusion-Generated Image Detection** Zhendong Wang, Jianmin.
