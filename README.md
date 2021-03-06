# FinalProject440
## Purpose
The goal of this project is to explore a new normalization method called regularized negative binomial regression ([Hafemeister 2019](https://www.biorxiv.org/content/10.1101/576827v1)) on building cell type classification models. 

## Repo Structure
### Data
This folder contains classification results for a merged 10K and 33K PBMC dataset from 10X Genomics. Data of interest is contained in files ending in _summary.csv.

### Scripts
This folder contains python scripts for creating the figure of choice for the homework.

## requirements.txt
This file contains python requirements for this project

## Creating a Figure
1. Ensure you are using Python 3.7
2. Ensure you have the correct requirements. From the project home folder (FinalProject440/) run:
```
pip install requirements.txt
```
3. Change File Permissions to Run Figure Script:
```
chmod +x runMeForFigure.sh
```
4. Run the Figure Script:
```
./runMeForFigure.sh
```

## Citations
Christoph Hafemeister, Rahul Satija
bioRxiv 576827; doi: https://doi.org/10.1101/576827

