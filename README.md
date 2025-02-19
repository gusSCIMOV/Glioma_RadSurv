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
See the ./config files 

### **General Seetings (main_config.ymal)**

Modify the dict entries accordingly. Enable the desired steps (by settimng True).

```yaml

#root_path: 'Y:/Groups/IDIAGroup/Data/_Brain/Radiology/_Adult/_Glioma' # data root path in your volume/system
root_path : ~./data # to use the project's data sample 

mri_sites:  # add as many dataset as nedeed (coded by site{n})
  site0: "UCSF-PDGM"
  site1: "LUMIERE"
  site2: "CCF"

mri_data: ["site1", "site2"] # data to be included in the pipeline

submodule_configs:
  preprocessing: "config/preprocessing_config.yaml"
  radiomics: "config/radiomics_config.yaml"

# Enabling modules
run_dicomSelection: True
run_dicomConversion: False
run_niftiSelection: False 
run_preprocessing: False
run_segmentation: False
run_radiomics: True

```
### **Preprocessing Seetings (preprocessing_config.ymal)**

```yaml
dirs:  # Modify the parent directories which store the preprocessing steps outputs and log files 
  raw_dicoms: 'dicom_raw'
  input_dicoms: 'dicom_structuralMRI'
  raw_nifti: 'nifti_raw'
  preprocessed: "nifti_preprocessed"
  metadata: "Metadata"

#pre-processign pipelines / parameters 
preprocessing_settings:
  query_nifti_file: 'niiQuery.csv' # {dataset}_niiQuery.csv contains the 4 MRI modalities to be included (see oen example at ~./data/TCGA-GBM/Metadata)
  acquisition_tag: 'Baseline' # time point to be analyzed
  query_key: 'included_modality' #df key inside of 
  mri_modalities: ["T1c","T1","T2","FLAIR"] # Strcutural MRI identifiers (strings. T1c might be also eferred as T1gd)

ants_preprocessing:
  first_step: 
    name: "Modality_2_atlas_reg"
    mri_str: "T1c" 
    atlas_str: "SRI24"
    ext: ".nii.gz"
    mask_str: None
    aff_metric : "mattes"
    type_of_transform : "Affine"
    transforming_mask: False
    brain_mask: None # HD_BET or Atlas_mask Default= None

  second_step: 
    name: "Modalities_coregistration"
    ext: ".nii.gz"
    aff_metric : "mattes"
    type_of_transform : "Affine"
    transforming_mask: False
    SkullStripp: None # HD_BET or Atlas_mask Default= None

  third_step: 
    name: "HD_SkullStripp"

  fourth_step: 
    name: "N4Bias_correction"
  fifth_step:  
    name: "IntensityScaling"

```



