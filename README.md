# Glioma_RadSurv
Radiomics and Machine learning pipeline for Survival Analysis

see the project_tree.txt to look at the project hierarchy

![pipeline](https://github.com/user-attachments/assets/f1f9a3d6-74f8-4c35-9b7e-7cc9e63c0ac3)

# Radiomics Risk Score (RRS) Pipeline

This repository contains a Python-based pipeline for computing a **Radiomics Risk Score (RRS)** for survival stratification in gliomas. The pipeline integrates morphological, textural, and biophysically informed signatures using multiple preprocessing and machine learning techniques.

## **Framework Overview**
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Modules Configuration](#modules-configuration)
- [Usage](#usage)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## **Project Structure**

Look at the project_tree.txt to see the folder hierarchy

## **Installation**
System Requeriments:

* Python =+3.9
* docker: nvcr.io/nvidia/pytorch:24.01-py3
* Ubuntu 22.04

To set up the project environment, follow these steps:

```sh
# Clone this repository
git clone https://github.com/gusSCIMOV/Glioma_RadSurv.git

# Change directory
cd Glioma_RadSurv

# Install dependencies
pip install -r requirements.txt
```

## **Modules Configuration**

