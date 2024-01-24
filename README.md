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
## Finetuning avec ResNet50

## Evauation & Expérimentation

## Conclusion
