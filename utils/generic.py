import os
import glob
import json
import logging
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import itertools
import shutil

from pathlib import Path
from datetime import datetime

class MRI_DataCheck:
    def __init__(self, mri_data,subdir_data,root_path, metadata_dir):
        self.mri_data=mri_data
        self.subdir=subdir_data
        self.root_path=root_path
        self.metadata_dir=metadata_dir
        self.logger = logging.getLogger(self.__class__.__name__) 
                             
    def find_patient_dirs(self,parent_dir, identifier):
        # identifier e.g, "TCGA-" "UCSF-PDGM"
        tcga_dirs = []

        # Walk through all directories recursively
        for root, dirs, files in os.walk(parent_dir):
            # Filter directories starting with "TCGA-"
            tcga_dirs.extend([d for d in dirs if d.startswith(identifier)])

        # Create dataframe with the found directories
        df = pd.DataFrame(tcga_dirs, columns=['ID_paths'])

        # Filter out entries ending with ".gz"
        df_filtered = df[~df['ID_paths'].str.endswith('.gz')]

        # Reset the index (optional)
        df_filtered.reset_index(drop=True, inplace=True)

        return df_filtered

    def browse_nifti_modalities(self,time_format='dates',summary=False, modalities_list=False, write_csv=True):
        """
        Creates a DataFrame from MRI directory structure with patient IDs and modalities.

        Args:
            root_dir (str): Path to the main directory containing patient subdirectories
        Returns:
            pd.DataFrame: DataFrame with patient IDs and corresponding MRI files summary or modalities list
        """
        if summary and modalities_list:
            self.logger.warning(f"summary and modality_list along, might be imcompatible")

        root_dir=os.path.join(self.root_path,self.mri_data,self.subdir)
        date_format = "%Y%m%d"
        data = []

        for patient_id in os.listdir(root_dir):
            patient_dir = os.path.join(root_dir, patient_id) # patient ID path
            if os.path.isdir(patient_dir):
                mri_studies=[f for f in os.listdir(patient_dir) 
                             if os.path.isdir(os.path.join(patient_dir,f))] # time points paths
                
                mri_studies=sorted(mri_studies) # sorting timepoints
                
                n_studies=len(mri_studies) # number of timepoints
                
                if time_format=='dates':
                    datetime_list = [datetime.strptime(date, date_format) for date in mri_studies]

                    intervals = [0] #The first position
                    intervals_name=['Baseline']

                    for i in range(1, len(datetime_list)):

                        days_difference = (datetime_list[i] - datetime_list[i - 1]).days
                        intervals.append(days_difference)
                        intervals_name.append(f"time{i:02}")

                    cumulative_dates=[]
                    cumulative_dates = list(itertools.accumulate(intervals))
                
                else:
                    datetime_list=mri_studies
                    intervals=mri_studies
                    intervals_name=mri_studies
                    cumulative_dates=mri_studies
                
                for timepoint, interval, interval_tag, date in zip(mri_studies, intervals, intervals_name, cumulative_dates):
                    timepoint_dir = os.path.join(patient_dir, timepoint)

                    nii_files = glob.glob(os.path.join(timepoint_dir, '*.nii.gz')) # mri modalities
                    n_mri = len(nii_files)
                    
                    if modalities_list:
                        for file in os.listdir(timepoint_dir):           
                            if file.endswith('.nii.gz'):
                                data.append({
                                    'ID': patient_id,
                                    'time_point': timepoint,
                                    'Acquistion_tag': interval_tag,
                                    'mri_modalities (original nifti)': file,
                                    'n_modalities_per_case': n_mri
                                })
                    
                        outstr=f"{self.mri_data}_{self.subdir}_modalities.csv"

                    if summary:         
                        data.append({
                                    'ID': patient_id,
                                    'time_point (date)': timepoint,
                                    'n_modalities_timepoint': n_mri,
                                    'acquisition_tag': interval_tag,
                                    'Acquistion interval (days)': interval,
                                    'Cumulative time from T-0 (days)': date,
                                    'n_time_points': n_studies
                                })
                        
                        outstr=f"{self.mri_data}_{self.subdir}_summary.csv"

        folders_df=pd.DataFrame(data)
        sorted_df = folders_df
        print(f"found {sorted_df.shape[0]} unique Case Ids")

        if write_csv:
            outfolder=os.path.join(self.root_path,self.mri_data,self.metadata_dir)
            os.makedirs(outfolder, exist_ok=True)
            outname=os.path.join(outfolder,outstr)
            sorted_df.to_csv(outname, index=False)
            print(f"saved as {outname}")
        else:
            print(f"enable write_csv=True to write a csv summary ")

        return sorted_df
    def to_construct_queryFile():
         """
         to construct a query file qwith the target modalities T1w, T1w-gd, T2w, FLAIR.
         """
    
    def copy_and_rename_files(self, pp_settings, source_dir, target_parent_dir):
        """
        Filters the DataFrame based on query arguments, copies the relevant MRI files from the source directory
        to the target directory while renaming them based on the `mri_tag` column.

        Args:
            dataframe (pd.DataFrame): The input DataFrame containing metadata for MRI files.
            query_args (dict): A dictionary where keys are column names and values are conditions for filtering the DataFrame.
            source_dir (str): The parent directory containing original MRI files.
            target_parent_dir (str): The parent directory where filtered and renamed files will be copied.

        Returns:
            None
        """
        # read query file (check the TRUE codigin inside the query_file.csv)
        base_path=os.path.join(self.root_path,self.mri_data)

        qfile_str=f"{self.mri_data}_{self.subdir}_modalities_{pp_settings.query_nifti_file}"
        qfile_path=os.path.join(base_path,self.metadata_dir,qfile_str)
        q_file=pd.read_csv(qfile_path)

        # Query arguments
        query_args = {
            'acquisition_tag': pp_settings.acquisition_tag,
            pp_settings.query_key: True
        }
        print(f"query {q_file}")

        # Filter the DataFrame based on the query arguments
        filtered_df = q_file.copy()
        for key, value in query_args.items():
            filtered_df = filtered_df[filtered_df[key] == value]
        
        print(f'filtered_df shape', {filtered_df.shape})

        # Iterate through the filtered DataFrame
        for _, row in filtered_df.iterrows():
            patient_id = row['ID']  # Assuming the column for Patient ID
            time_point = row['time_point']  # Acquisition tag like Baseline, Timepoint 01, etc.
            mri_tag = row['mri_tag']  # The new name for each file
            filename=row["mri_modalities (original nifti)"]
            new_filename = f"{patient_id}_{mri_tag}.nii.gz"
            # Construct the source and target directories
            source_path = os.path.join(base_path,source_dir , 
                                       str(patient_id) , str(time_point))
            target_path = os.path.join(base_path,target_parent_dir,
                                       str(patient_id) , str(pp_settings.acquisition_tag))
            
            if not os.path.isdir(source_path):
                print(f"Source path does not exist: {source_path}")
                continue
            
            # Create the target directory if it doesn't exist
            os.makedirs(target_path, exist_ok=True)
            
            file_path=os.path.join(source_path, filename)
            target_file = os.path.join(target_path, new_filename)
            shutil.copy2(file_path, target_file)  # Copy with metadata preserved
            print(f"Copied and renamed: {file_path} -> {target_file}")


    def print_folder_tree(self, max_depth=2):
        root_path=os.path.join(self.root_path,self.mri_data,self.subdir)
        root = Path(root_path)
    
        def print_tree(path, prefix="", depth=0):
            if depth > max_depth:
                return

            print(f"{prefix}├── {path.name}")

            if depth < max_depth:
                items = sorted(list(path.iterdir()))
                dirs = [item for item in items if item.is_dir()]

                for dir in dirs[:-1]:
                    print_tree(dir, prefix + "│   ", depth + 1)
                if dirs:
                    print_tree(dirs[-1], prefix + "    ", depth + 1)

            #print(f"📁 {root.name}")
        print_tree(root)

    def CrossCheckDF(self, df_list, df_indx, write_csv=False):
        # Create a list of series from each dataframe's patient ID column
        series_list = []
        for i, df in enumerate(df_list):
            # Assuming the ID column is the first/only column
            df_size=df.shape
            series = pd.Series(df.iloc[:, 0].values, index=df.iloc[:, 0].values, name=f"{df_indx[i]}_n_{df_size[0]}")
            series_list.append(series)

        # Merge all series using outer join on the index
        # This will align matching TCGA IDs in the same row
        merged_df = pd.concat(series_list, axis=1)

        # Sort the index (TCGA IDs) if needed
        merged_df = merged_df.sort_index()
        
        outstr=f"{self.mri_data}_cross_check_temp.csv"
        
        if write_csv:
            outname=os.path.join(self.root_path,
                                          self.mri_data,'Metadata',outstr)
            merged_df.to_csv(outname, index=False)
            print(f"saved as {outname}")
        else:
            print(f"enable write_csv=True to write a csv summary ")
            
        return merged_df



def getting_image_list(input_dir,time_point,mri_mod,ext,mask_str, pipeline=''):
    #root_path = '/app/Data/_Brain/Radiology/_Adult/_Glioma/'
    main_path=os.path.join(input_dir,'*',time_point, '*'+ mri_mod + pipeline + ext)
    #print('main_path',main_path)
    imgs = glob.glob(main_path)
    imgs=sorted(imgs)
    
    if mask_str != None:
        main_path=os.path.join(input_dir,'*',time_point, '*' + mask_str +'.nii*')
        labels = glob.glob(main_path)
        labels=sorted(labels)
        #print('\n number of segmentation files: {}'.format(len(labels)))
    else:
        labels=mask_str
    
    return imgs, labels

def getting_images_list_from_csv(root_path,mri_data,subdir,mri_mod,file_xlsx_list,df_key, mask_str, 
                                ID_filtering=False, time_filtering=False, time_point=None, pipeline=None, print_search=False):
    
    xlsx_path=os.path.join(root_path,mri_data,'Metadata',f"{mri_data}_{file_xlsx_list}")
    data_id=pd.read_csv(xlsx_path, dtype={0: str})
    
    ID = data_id['ID1'].tolist()
    segs=[]
    imgs=[]
    
    for ids in ID:

        if time_filtering:
            tp = data_id.loc[data_id['ID1'] == ids, df_key].values[0]
        elif ID_filtering:
            tp=time_point

        if pipeline==None:
            img_paths=os.path.join(root_path , mri_data, subdir, ids, tp, '*_' + mri_mod + '.nii*')
        else:
            img_paths=os.path.join(root_path , mri_data, subdir, ids, tp, '*_' + mri_mod + '*' + pipeline +'.nii*')

        if print_search:
            print("img_paths", img_paths)
        
        img_file=glob.glob(img_paths)
        if len(img_file)==0:
            continue
        else:
            imgs.append(img_file[0])
        
        seg_path=os.path.join(root_path, mri_data, subdir, ids, tp, ids +'_' + '*' + mask_str +'.nii*')
        if print_search:
            print("img_paths", seg_path)

        seg_file=glob.glob(seg_path)

        if len(seg_file)==0:
            continue
        else:
            segs.append(seg_file[0])

    imgs=sorted(imgs)
    segs=sorted(segs)
    print('\n number of images files: {}, segmentation files: {} for MRI modality: {}, from : {} '.format(len(imgs), len(segs), mri_mod ,xlsx_path))
    return segs, imgs, data_id


def Pairing_Label_mri(file_path, mov_imgs,  mri_mod, list_ref_imgs=None, ID_level=-3):
    
    id_label=Path(file_path).parts[ID_level]
    print('\nPatient ID: .. {}'.format(id_label))
    
    mri_idx = [idx for idx, fname in enumerate(mov_imgs) if f"{id_label}_{mri_mod}" in fname]
    
    if len(mri_idx) != 0:
        mov_im=mov_imgs[mri_idx[0]]
        print("mov MRI scan: .. {}".format(mov_im))
    elif len(mri_idx) == 0:
        mov_im=None
        print('{} mri modality does not exist for {}'.format(mri_mod,id_label))


    if list_ref_imgs!=None:
        mri_idx = [idx for idx, fname in enumerate(list_ref_imgs) if f"{id_label}_{mri_mod}" in fname]
        
        if len(mri_idx) != 0:
            ref_im=list_ref_imgs[mri_idx[0]]
            print("Ref MRI scan: .. {}".format(ref_im))
        elif len(mri_idx) == 0:
            ref_im=None
            print('{} mri modality does not exist for {}'.format(mri_mod,id_label))
        
    return  id_label, mov_im, ref_im

def Pairing_mri(mov_img, ref_imgs, ID_level=-3):
    
    id_label=Path(mov_img).parts[ID_level]
    print('\nPatient ID: .. {}'.format(id_label))
    
    mri_idx = [idx for idx, fname in enumerate(ref_imgs) if f"{id_label}" in fname]
    
    if len(mri_idx) != 0:
        ref_im=ref_imgs[mri_idx[0]]
        print("REF MRI scan: .. {}".format(ref_im))
    elif len(mri_idx) == 0:
        ref_im=None
        print('{} mri modality does not exist for {}'.format(id_label))

        
    return  id_label, ref_im

def safe_log(msg):
    return msg.encode("ascii", "ignore").decode()  # Removes non-ASCII characters for loggers


    