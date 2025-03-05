import os
import logging
import pandas as pd

from preprocessing.dicom_handling import DicomConversion # class
from utils.config_loader import *
from preprocessing.antspy_pp_utils import *
from preprocessing.Select4modalities import *
from utils.generic import *

logger = logging.getLogger(__name__)

def MRI_preprocessing(configs, project_path):

    gen_config=Config2Struct(configs["general"])
    gen_config.project_path=project_path
    root_path = gen_config.root_path
    mri_data=gen_config.mri_data

    dirs = Config2Struct(configs["preprocessing"]["dirs"])
    pp_settings = Config2Struct(configs["preprocessing"]["preprocessing_settings"])

    filtered_sites=[dataset for key , dataset in gen_config.mri_sites.items() if key in mri_data] # setting target datasets
    
    for dataset in filtered_sites:

        id_tag=dataset
        log_path=os.path.join(root_path,dataset,dirs.logs)

        if gen_config.run_dicomSelection:
            
            logger.info(f"{dataset} Starting dicom selection")

            raw_dcm_path=os.path.join(root_path,dataset,dirs.raw_dicoms)
            dcm_path=os.path.join(root_path,dataset,dirs.input_dicoms)

            DicomSelection(raw_dcm_path, dcm_path, log_path, n_studies=1) # from ./preprocessing.Select4modalities


        if gen_config.run_dicomConversion:  # running dicom to nifti conversion (set input and output directory)
                
            logger.info(f"{dataset} Starting dicom conversion")
                    
            dcm_path=os.path.join(root_path,dataset,dirs.input_dicoms)
            nii_gz_path=os.path.join(root_path,dataset,dirs.raw_nifti)
            dcm_converter=DicomConversion(dcm_path, nii_gz_path, id_tag)
            dcm_converter.convert_dataset_to_nifti()


            MRIQc=MRI_DataCheck(dataset, dirs.raw_nifti, root_path, dirs.logs) # from utils.generic
            MRIQc.browse_nifti_modalities(time_format='dates',modalities_list=True, write_csv=True)
            MRIQc.browse_nifti_modalities(time_format='dates',summary=True, write_csv=True)

            logger.info(f" DICOM CONVERSION DONE FOR {dataset}")

        if gen_config.run_niftiSelection: # run quality check to retrieve the structural modalities:

            logger.info(f" Starting Nifti selection for {dirs.raw_nifti} ")

            
            MRIQc=MRI_DataCheck(dataset, dirs.raw_nifti, root_path, dirs.logs) # from utils.generic
            MRIQc.copy_and_rename_files(pp_settings, dirs.raw_nifti, dirs.preprocessed)
            
            logger.info(f" Nifti SELECTION DONE for {dataset}")


        if gen_config.run_preprocessing:  # set the customized pipeline (preprocessing_config.yaml)

            logger.info(f" Starting MRI Preprocessing for {dataset}")

            pp_config = Config2Struct(configs["preprocessing"]["ants_preprocessing"])
            #get_config_params(pp_config) # print Config attributtes 

            MRIpp=MRIPreprocessing(dataset, dirs, gen_config, pp_settings, pp_config) # from preprocessing.ants_pp_utils
            
            MRIpp.run_pipeline()

                         
            # if pp_config.xlsx_file != None:

            #     labels, imgs_list = getting_image_list(root_path, mri_data,subdir,time_point,mri_mod,pipeline,ext,mask_str)
            # else:
            #     labels_imgs_list  = getting_images_list_from_csv(root_path,mri_data,subdir,mri_mod,file_xlsx_list,df_key, mask_str, 
            #                     ID_filtering=False, time_filtering=False, time_point=None, pipeline=None, print_search=False)

                



            