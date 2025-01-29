import os
import logging
import pandas as pd

from utils.config_loader import *
from preprocessing.dicom_handling import DicomConversion # class
from preprocessing.antspy_pp_utils import *
from utils.generic import *

logger = logging.getLogger(__name__)

def MRI_preprocessing(configs, project_path):

    config=Config2Struct(configs["general"])
    root_path = config.root_path
    mri_data=config.mri_data

    dirs = Config2Struct(configs["preprocessing"]["dirs"])

    filtered_sites=[dataset for key , dataset in config.mri_sites.items() if key in mri_data] # setting target datasets
    
    for dataset in filtered_sites:

        id_tag=dataset
        
        if config.run_dicomConversion:  # running dicom to nifti conversion (set input and output directory)
                
            logger.info(f"✅ Starting dicom conversion")
                    
            dcm_path=os.path.join(root_path,dataset,dirs.input_dicoms)
            nii_gz_path=os.path.join(root_path,dataset,dirs.raw_nifti)
            dcm_converter=DicomConversion(dcm_path, nii_gz_path, id_tag)
            dcm_converter.convert_dataset_to_nifti()


            MRIQc=MRI_DataCheck(dataset, dirs.raw_nifti, root_path, dirs.metadata)
            MRIQc.browse_nifti_modalities(time_format='dates',modalities_list=True, write_csv=True)
            MRIQc.browse_nifti_modalities(time_format='dates',summary=True, write_csv=True)

            logger.info(f"✅ DICOM CONVERSION DONE FOR {dataset}")

        if config.run_niftiSelection: # run quality check to retrieve the structural modalities:

            logger.info(f"✅ Starting Nifti selection for {dirs.raw_nifti} ")

            pp_settings = Config2Struct(configs["preprocessing"]["preprocessing_settings"])
            MRIQc=MRI_DataCheck(dataset, dirs.raw_nifti, root_path, dirs.metadata)
            MRIQc.copy_and_rename_files(pp_settings, dirs.raw_nifti, dirs.preprocessed)
            
            logger.info(f"✅ Nifti SELECTION DONE for {dataset}")


        if config.run_preprocessing:  # set the customized pipeline (preprocessing_config.yaml)

            logger.info(f"✅ Starting MRI Preprocessing")

            pp_config = Config2Struct(configs["preprocessing"]["ants_preprocessing"])
            get_config_params(pp_config) # print Config attributtes 

            for key, value in vars(pp_config).items():
                print(f"{key} {value}")

            
            #atlas_path= project_path +'/preprocessing/ATLAS_T1/'+ atlas_str +'/templates/T1_brain.nii'
             
            # if pp_config.xlsx_file != None:

            #     labels, imgs_list = getting_image_list(root_path, mri_data,subdir,time_point,mri_mod,pipeline,ext,mask_str)
            # else:
            #     labels_imgs_list  = getting_images_list_from_csv(root_path,mri_data,subdir,mri_mod,file_xlsx_list,df_key, mask_str, 
            #                     ID_filtering=False, time_filtering=False, time_point=None, pipeline=None, print_search=False)

                



            