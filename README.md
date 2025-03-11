# DarwinCoreToDataSetTools
Tools for creating datasets using observations with images from a Darwin Core. Tested on Python 3.11.11

## Description 
This repository contains a pipeline for compiling herbarium specimen image datasets using images sourced from the Global Biodiversity Information Facility (GBIF). By streamlining dataset creation, this approach supports applications in taxonomy, herbarium curation, and biodiversity research—bridging the gap between digital and physical herbarium collections. The pipeline processes images and observations formatted in DarwinCore (DwC), the biodiversity data standard, addressing key challenges in dataset creation.


The pipeline includes methods for: 

<ul>
<li>Balanced Dataset Generation: Mitigating data imbalance by limiting over-represented taxa and ensuring sufficient representation of under-represented taxa</li>
<li>Data Cleaning & Filtering:Detecting and removes corrupted images</li> 
<li>Feature Removal Blurs non-plant features (e.g., text labels, logos, barcodes)</li>
<li>Dataset Structuring: Organizes images into train/test sets for robust model evaluation</li>
<li>Model Training: Training and finetuning of a ViT classifier</li>
</ul>

## Example
See Example.ipynb

