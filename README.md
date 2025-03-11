# DarwinCoreToDataSetTools
Tools for creating datasets using observations with images from a Darwin Core. Tested on Python 3.11.11

## Description 
This repository contains a pipeline for compiling herbarium specimen image datasets using images sourced from the Global Biodiversity Information Facility (GBIF). By streamlining dataset creation, this approach supports applications in taxonomy, herbarium curation, and biodiversity research—bridging the gap between digital and physical herbarium collections. The pipeline processes images and observations formatted in DarwinCore (DwC), the biodiversity data standard, addressing key challenges in dataset creation.

The pipeline includes methods for: (1) **Balanced Dataset Generation:** Mitigates data imbalance by limiting over-represented taxa and ensuring sufficient representation of under-represented taxa; (2) **Data Cleaning & Filtering:** Detects and removes corrupted images; (2) **Feature Removal:** Blurs non-plant features (e.g., text labels, logos, barcodes); (3) **Dataset Structuring:** Organizes images into train/test sets for robust model evaluation; (4) **Model Training:** Training and finetuning of a ViT classifier. 

## Example
See Example.ipynb

