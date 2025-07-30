# DarwinCoreToDataSetTools
Tools for creating datasets using observations with images from a Darwin Core. Tested on Python 3.11.11

## Description 
This repository contains a pipeline for compiling herbarium specimen image datasets using images sourced from the Global Biodiversity Information Facility (GBIF). By streamlining dataset creation, this approach supports applications in taxonomy, herbarium curation, and biodiversity research—bridging the gap between digital and physical herbarium collections. The pipeline processes images and observations formatted in DarwinCore (DwC), the biodiversity data standard, addressing key challenges in dataset creation.


The pipeline includes methods for: 

<ul>
<li>Balanced dataset generation for mitigating data imbalance by limiting over-represented taxa and ensuring sufficient representation of under-represented taxa</li>
<li>Data cleaning & filtering for detecting and removes corrupted images</li> 
<li>Feature removal to blurs non-plant features (e.g., text labels, logos, barcodes)</li>
<li>Dataset structuring to organise images into train/test sets for robust model evaluation</li>
<li>Model training methods for training and finetuning of a ViT classifier</li>
</ul>

## Example
See Example.ipynb

To use this example the model weights linked below need to be downloaded and placed in the folder label_blurring/CRAFT_models

## CRAFT weights
Official github: https://github.com/clovaai/CRAFT-pytorch
craft_mlt_25 weights: https://drive.google.com/file/d/1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ/view

