# GROVER  
Graph‑guided Representation of Omics and Vision with Expert Regulation for Adaptive Spatial Multi‑omics Fusion  
![Framework](https://github.com/endingcoding/GROVER/blob/main/GROVER/Framework.jpg)

## Prerequisite  
* Python 3.10  
* torch 2.0.1+cu118  
* torchvision 0.16.1+cu118  
* torchaudio 2.1.1+cu118  
* torch-geometric 2.6.1  
* torch-scatter 2.1.2  
* torch-sparse 0.6.18  
* numpy 1.26.3  
* scipy 1.14.0  
* scikit-learn 1.5.1  
* pandas 2.2.2  
* matplotlib 3.9.1  
* seaborn 0.13.2  
* tqdm 4.66.4

## Getting Started

### Datasets
We used the following 10x Genomics Visium CytAssist datasets:

- **Human Breast Cancer, IF, 6.5 mm (FFPE)**  
  [Gene and Protein Expression Library](https://www.10xgenomics.com/datasets/gene-and-protein-expression-library-of-human-breast-cancer-cytassist-ffpe-2-standard)

- **Human Glioblastoma, IF, 11 mm (FFPE)**  
  [Gene and Protein Expression Library](https://www.10xgenomics.com/datasets/gene-and-protein-expression-library-of-human-glioblastoma-cytassist-ffpe-2-standard)

- **Human Tonsil, H&E, 6.5 mm (FFPE)**  
  [Gene and Protein Expression Library](https://www.10xgenomics.com/datasets/gene-protein-expression-library-of-human-tonsil-cytassist-ffpe-2-standard)

- **Human Tonsil with Add-on Antibodies, H&E, 6.5 mm (FFPE)**  
  [Gene and Protein Expression Library](https://www.10xgenomics.com/datasets/visium-cytassist-gene-and-protein-expression-library-of-human-tonsil-with-add-on-antibodies-h-e-6-5-mm-ffpe-2-standard)

---

### Obtaining Image Embeddings (`img_emb`)
Due to privacy restrictions on **Omiclip** weights, you need to contact the authors of [Omiclip](https://www.nature.com/articles/s41592-025-02707-1) to obtain the pretrained model. Follow the Omiclip tutorial to extract `img_emb`, which can then be directly used in GROVER.  

> Note: GROVER is flexible—any advanced pathology large model that outputs image embeddings can be used as input.

