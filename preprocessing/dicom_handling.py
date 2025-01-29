import os
import re
import pydicom
import dicom2nifti
import shutil
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict

class DicomConversion:
    def __init__(self, dcm_path, nii_gz_path, id_tag):
        self.dcm_path = dcm_path
        self.nii_gz_path = nii_gz_path
        self.id_tag=id_tag
        self.logger = logging.getLogger(self.__class__.__name__) 

    def group_multicomponent_dicoms(self, dicom_files_paths):
        """
        Group DICOM files by their component identifier.
        Checks common tags that distinguish different components in a series.
        
        Args:
            dicom_files_paths (list): List of paths to DICOM files
        
        Returns:
            dict: Components grouped by their identifying characteristics
        """
        components = defaultdict(list)
        
        for dcm_path in dicom_files_paths:
            try:
                dcm = pydicom.dcmread(dcm_path, force=True)
                
                # Create a tuple of identifying characteristics
                # Add or modify these based on your specific DICOM metadata
                component_key = (
                    getattr(dcm, 'EchoTime', None)
                )
                
                components[component_key].append(dcm_path)
                
            except Exception as e:
                print(f"   Warning: Could not read DICOM file {dcm_path}: {str(e)}")
                continue
        
        return components

    def get_dicom_date(self, dicom_data):
        """
        Try to get the date from various DICOM tags in order of preference.
        
        Args:
            dicom_data: PyDICOM dataset
        
        Returns:
            formatted_date (str): Formatted date string or None if no date found
        """
        date_attributes = [
            'AcquisitionDate',
            'SeriesDate',
            'StudyDate'
        ]
        
        for attr in date_attributes:
            try:
                date_value = getattr(dicom_data, attr)
                if date_value:
                    return datetime.strptime(date_value, '%Y%m%d').strftime('%Y%m%d')
            except (AttributeError, ValueError):
                continue
        
        return None

    def try_alternative_conversion(self):
        """
        Attempt alternative conversion methods when dicom2nifti fails.
        This is a placeholder for implementing alternative conversion methods.
        """
        # TODO: Implement alternative conversion methods
        # For example, using SimpleITK, nibabel, or other libraries
        return False

    def convert_dataset_to_nifti(self, onset=0):
        """
        Convert DICOM dataset to NIfTI format with multi-component handling.
        """

        Path(self.nii_gz_path).mkdir(parents=True, exist_ok=True)
        timepoint_counters = {}
        
        IDlist=os.listdir(self.dcm_path)
        IDlist=sorted(IDlist)
        sublist=(IDlist[onset:])
        print(f"batch size {len(sublist)}")
        
        for patient_folder in sublist:
            patient_path = os.path.join(self.dcm_path, patient_folder)
            
            if not os.path.isdir(patient_path):
                continue
                
            print("\n" + "="*50)
            print(f"Processing patient: {patient_folder}")
            print("="*50 + "\n")
                
            output_patient_path = os.path.join(self.nii_gz_path, patient_folder)
            Path(output_patient_path).mkdir(parents=True, exist_ok=True)
            
            if patient_folder not in timepoint_counters:
                timepoint_counters[patient_folder] = 1
            
            for timepoint_folder in os.listdir(patient_path):
                timepoint_path = os.path.join(patient_path, timepoint_folder)
                
                if not os.path.isdir(timepoint_path):
                    continue
                    
                print(f"--> Timepoint: {timepoint_folder}")
                
                for series_folder in os.listdir(timepoint_path):
                    series_path = os.path.join(timepoint_path, series_folder)
                    
                    if not os.path.isdir(series_path):
                        continue
                        
                    try:
                        # Check for DICOM files at the third level
                        if not self.has_subfolders(os.path.join(series_path)):
                            dicom_files = [os.path.join(series_path, f) 
                                        for f in os.listdir(series_path) 
                                        if f.endswith('.dcm')]
                        # Check for a fourth level and process DICOM files there   
                        elif self.has_subfolders(os.path.join(series_path)):
                            for subfolder in os.listdir(series_path):
                                subfolder_path = os.path.join(series_path, subfolder)
                                dicom_files = [os.path.join(subfolder_path, f)
                                            for f in os.listdir(subfolder_path) 
                                            if f.lower().endswith(".dcm")]
                        
                        if dicom_files:
                            print(f"----> DICOM files found at third level in: {os.path.basename(series_path)}")

                        elif not dicom_files:
                            print(f"   No DICOM files found in {os.path.basename(series_path)}")
                            continue
                        
                        first_dicom = pydicom.dcmread(dicom_files[0])
                        
                        try:
                            series_description = first_dicom.SeriesDescription
                            series_description = "".join(x for x in series_description 
                                                    if x.isalnum() or x in "_ -").strip()
                        except AttributeError:
                            series_description = f"series_{series_folder}"
                            print(f"   Warning: No SeriesDescription found for {series_folder}")
                        
                        dicom_date = self.get_dicom_date(first_dicom)
                        
                        if dicom_date:
                            output_timepoint_path = os.path.join(output_patient_path, dicom_date)
                            print(f"   Using date from DICOM tags: {dicom_date}")
                        else:
                            timepoint_name = f"time{timepoint_counters[patient_folder]:02d}"
                            output_timepoint_path = os.path.join(output_patient_path, timepoint_name)
                            print(f"   No date found in DICOM tags, using {timepoint_name}")
                            timepoint_counters[patient_folder] += 1
                        
                        Path(output_timepoint_path).mkdir(parents=True, exist_ok=True)
                        
                        try:
                            # First attempt: standard conversion
                            output_nifti = os.path.join(output_timepoint_path, 
                                                    f"{series_description}.nii.gz")
                            
                            if not os.path.exists(output_nifti):
                                dicom2nifti.dicom_series_to_nifti(series_path, 
                                                                output_nifti,
                                                                reorient_nifti=True)
                                print(f"   Successfully converted: {output_nifti}")
                            else:
                                print(f"   Nifti volume Already exist : {os.path.basename(output_nifti)}")
                                
                        except Exception as e:
                            if "NOT_A_VOLUME" in str(e):
                                print(f"   Detected multi-component series: {series_description}")
                                print("   Attempting to separate components...")
                                
                                # Group DICOM files by component
                                components = self.group_multicomponent_dicoms(dicom_files)
                                #print(f"components : {(components.items())}")
                                
                                for idx, (comp_key, comp_files) in enumerate(components.items(), 1):
                                    if len(comp_files) > 1:  # Only process if we have multiple files
                                        temp_dir = os.path.join(series_path, f'temp_component_{idx}')
                                        Path(temp_dir).mkdir(exist_ok=True)
                                        
                                        # Copy files to temporary directory
                                        for f in comp_files:
                                            shutil.copy2(f, temp_dir)
                                        
                                        # Try to convert this component
                                        try:
                                            output_nifti = os.path.join(
                                                output_timepoint_path,
                                                f"{series_description}_component{idx}.nii.gz"
                                            )
                                            
                                            if not os.path.exists(output_nifti):
                                                dicom2nifti.dicom_series_to_nifti(
                                                    temp_dir,
                                                    output_nifti,
                                                    reorient_nifti=True
                                                )
                                                print(f"   Successfully converted component {idx}: {output_nifti}")
                                        
                                        except Exception as comp_e:
                                            print(f"   Error converting component {idx}: {str(comp_e)}")
                                            
                                        # Clean up temporary directory
                                        shutil.rmtree(temp_dir)
                                
                            else:
                                print(f"   Error converting series {series_description}: {str(e)}")
                                
                    except Exception as e:
                        print(f"   Error processing series {series_folder}: {str(e)}")
                        continue
                        
        print("\n" + "="*50)
        print("Dataset Conversion (nifti) complete!")
        print("="*50)



    def has_subfolders(self,folder_path):
        for item in os.listdir(folder_path):
            if os.path.isdir(os.path.join(folder_path, item)):
                return True
        return False
    
    def verify_conversion(self):
        """
        Verify that all DICOM series were converted to NIfTI.
        
        Args:
            self.dcm_path (str): Original DICOM dataset path
            self.nii_gz_path (str): Converted NIfTI dataset path
        """
        input_count = 0
        output_count = 0
        
        # Count DICOM series
        for root, dirs, files in os.walk(self.dcm_path):
            if any(f.endswith('.dcm') for f in files):
                input_count += 1
        
        # Count NIfTI files
        for root, dirs, files in os.walk(self.nii_gz_path):
            output_count += len([f for f in files if f.endswith('.nii.gz')])
        
        print(f"Total DICOM series: {input_count}")
        print(f"Total NIfTI files created: {output_count}")
        print(f"Conversion rate: {(output_count/input_count)*100 if input_count > 0 else 0:.2f}%")
        