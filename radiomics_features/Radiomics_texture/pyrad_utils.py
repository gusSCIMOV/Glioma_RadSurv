# gpinedaortiz
import os
import glob
import numpy as np
import SimpleITK as sitk
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
import nibabel as nib
import psutil

from functools import reduce
from radiomics import featureextractor
from matplotlib.gridspec import GridSpec
from radiomics import firstorder, glcm, imageoperations, shape, glrlm, glszm
from matplotlib.font_manager import FontProperties
from scipy.ndimage import zoom
from joblib import Parallel, delayed

from pathlib import Path



class FilePath_tags:
    def __init__(self, **kwargs):
        self.params = kwargs
        self.count=0
    
    def data_counter(self):
        self.count += 1
        return self.count

    def input_tags(self):
        print("Segmentation with BraTS convention: necrosis NEC : 1, Edema Ed : 2, Enhancing ET : 4 , All region TM : 5 ")
        print("\nConstructing file paths / lists")
        self.label_val=self.params['label_val']
        self.mri_mod=self.params['mri_mod']
        self.habitat=self.params['habitats'][self.params['habitat']]
        
        if self.habitat == 'Enhancing':
            self.sufx='ET'
        elif self.habitat == 'Edema': 
            self.sufx='ED'
        elif self.habitat == 'All_tumor': 
            self.sufx='TM'
            
        feat=self.params['feature_family'][self.params['feature']]
        self.name_a = f'{self.habitat}_{feat}'
        print("habitat: {}, sufix: {}, feature name: {}, label in mask: {}".format(self.habitat, self.sufx, self.name_a, self.label_val))

        self.root_path=self.params['root_path']
        self.mri_data=self.params['mri_sites'][self.params['mri_data']]

        subdir=self.params['subdirs'][self.params['subdir_data']]
        self.time_point=self.params['time_point'][self.params['tp']]
        mask_str=self.params['masks'][self.params['mask_str']]
        self.pipeline=self.params['pipelines'][self.params['pipeline']]

        labels_path=os.path.join(self.root_path , self.mri_data, subdir,'*',self.time_point, '*' + mask_str +'.nii.gz')
        labels = glob.glob(labels_path)
        self.labels = sorted(labels)

        main_path=os.path.join(self.root_path , self.mri_data, subdir,'*',self.time_point, '*_'+ self.mri_mod + '*' + self.pipeline +'.nii*')
        imgs = glob.glob(main_path)
        self.imgs=sorted(imgs)
        print('\nNumber of segmentation files: {}, number of MRI scans: {} , for MIR modality: {}'.format(len(labels), len(imgs), self.mri_mod))
        
        self.output_fmatrix=self.params['output_fmatrix'] 
        
        return self.habitat, self.sufx, self.name_a, self.labels, self.imgs, self.label_val

    def Pairing_Label_mri(self,file_path, ID_level=-3):
        self.counter=self.data_counter()
        self.id_label=Path(file_path).parts[ID_level]
        print('\nPatient ID: .. {}, time point:{}'.format(self.id_label,self.time_point))
        
        mri_idx = [idx for idx, fname in enumerate(self.imgs) if self.id_label + '_' + self.mri_mod in fname]
        
        if len(mri_idx) != 0:
            print("MRI scan: .. {}".format(self.imgs[mri_idx[0]]))
        elif len(mri_idx) == 0:
            print('{} mri modality does not exist for {}'.format(self.mri_mod,self.id_label))
            
        return  self.id_label, mri_idx
    
    
    def root_out_folders(self):
        output_features=self.params['output_features']
        self.outpath_base=os.path.join(self.root_path,self.mri_data,output_features)
#         print("outpath_base : {}".format(self.outpath_base))
        
        self.feat_folder=self.params['feature_folders'][self.params['feat_folder']]
   
        self.output_name=os.path.join(self.outpath_base , self.id_label , self.time_point ,self.feat_folder)
    
        if not os.path.exists(self.output_name):
                os.makedirs(self.output_name)
        
        return self.outpath_base, self.output_name
    
    
    def Output_tags(self,ws,binWidth,levels):
        
        _, self.output_name=self.root_out_folders()
        
        self.ws=ws
        self.binWidth=binWidth
        self.levels=levels
    
        tmp_name =  os.path.join(self.output_name, self.mri_mod + '_' + self.id_label +'_'+ 
                                 self.name_a + '_ws_'+str(ws)+'_'+ str(levels/binWidth)+'_' + self.pipeline + '_stats.npy') 
        
        tmp_name_extractor =  tmp_name.replace('_stats.npy','_extractor.npy')
        
        return tmp_name, tmp_name_extractor
    
    def processing_logger(self):
        
        port=self.params['port']
        str_log = self.id_label +'_'+ self.name_a + '_ws_'+str(self.ws)+'_'+ str(self.levels/self.binWidth)+ ' MRI SCAN No '+ str(self.counter)+'/'+str(self.batch_size)
        log_file = os.path.join(
            self.outpath_base,self.mri_mod +'_'
            + self.name_a +'_batch_'+ self.batch +
            '_port_'+ port +
            '_progress_status.txt')
        
        return str_log , log_file
    

    def batch_split(self,print_batch=False):
        self.batch_size=self.params['batch_size']
        self.batch=self.params['batch']
        id1=self.params['id1']
        id2=self.params['id1']+self.batch_size
        
        batch_list=list(self.labels[id1:id2])
        if print_batch:
            print("\nBatch size: {}".format(self.batch_size))
            for image in batch_list:
                print(image)

        return batch_list

    def getting_images_list_from_csv(self, mri_data,subdir,time_point,mri_mod,pipeline,mask_str):

        # read list of patients from a list CSV XLSX
        xlsx_path=root_path+'UCSF/UCSF-PDGM/Metadata/UCSF-Pre-Post-Os.xlsx'
        data_id=pd.read_excel(xlsx_path)
        ID=data_id['UCSF-PDGM_ID'].tolist()

        labels=[]
        for ids in ID:
            mask_file=glob.glob(os.path.join(root_path , mri_data, mri_data + '-PDGM' '/Preprocessed/*/time02/',ids +'*seg_brats.nii.gz'))
            #print(os.path.join(root_path , mri_data, mri_data + '-PDGM' '/Preprocessed/Post-Op/*/time01/',ids +'*seg_brats.nii.gz'))
            labels.append(mask_file[0])
            #print(mask_files)

        labels=sorted(labels)
        len(labels)
        return labels

    def list_StatsFiles(self):
        # variables newly generated and non accesibles by self.root_out_folders()
        
        output_features=self.params['output_features']
        outpath_base=os.path.join(self.root_path,self.mri_data,output_features)
        feat_folder=self.params['feature_folders'][self.params['feat_folder']]
        
        npy_files = os.path.join(outpath_base ,'*', self.time_point , feat_folder)
        npy_files=glob.glob(npy_files)
        self.npy_files=sorted(npy_files)
        print('N {} {} folders ....... '.format(len(self.npy_files),feat_folder))
    
        return self.npy_files

class Wrap_Features:

    def __init__(self, **kwargs):
        self.params = kwargs
        
    def glcm_fmatrix(self,kernel_radius,binWidth_ranges,levels, FileTags, ID_level=-3, print_ID=False):
        # FilePath_tags: an instance of FilePath_tags.input_tags() method
        self.kernel_radius=kernel_radius
        self.binWidth_ranges=binWidth_ranges
        self.dim=len(binWidth_ranges)*len(kernel_radius)
        self.bines = list(map(lambda x: levels / x, binWidth_ranges))
        
        h_names, feat_names, mri_tag = self.glcm_names(FileTags)
        feature_dim=len(h_names)*4 # feature vect dim (n of glcm times 4 statistics ) per scale (each ws and bin pair)

        id_list=[]
        feature_mat=[]
        for subject in list(FileTags.npy_files[:]):

            id_label = Path(subject).parts[ID_level]
            test_files=[]

            for filename in os.listdir(subject):
                if FileTags.mri_mod in filename:
                    test_files.append(filename)

            if not test_files or not any(FileTags.habitat in file_name for file_name in test_files):
                print("\nID: {} Non existing {} or Empty directory for {}".format(id_label,FileTags.habitat , FileTags.mri_mod))
                continue

            
            feat_vect1=[]
            for ws in kernel_radius:
                for bins in self.bines:
                    feature_file=os.path.join(subject, FileTags.mri_mod+'_'+id_label+'_'+FileTags.name_a+'_ws_'+str(ws)+'_'+str(bins)+ '*stats.npy')
                    npy_file = glob.glob(feature_file)
        
                    if len(npy_file)==1:
                        feat_vect=np.load(npy_file[0])
                        if feat_vect.shape[1]==feature_dim:
                            feat_vect1.append(feat_vect) 
                        else:
                            print("Upcomming error: current feature dimension: {}, expected: {}".format(feat_vect.shape[1],feature_dim))
                            raise ValueError("invalid feature dimension for: window size: {} and, bin: {} levels for ID {}".format(ws,bins,id_label))

                    elif not npy_file:
                        raise ValueError("missing features for: window size: {} and bin: {} levels for ID {}".format(ws,bins,id_label))
            
            feat_vect1=np.squeeze(feat_vect1) 
            
            feat_vect1=feat_vect1.reshape(1,self.dim*feat_vect.shape[1])   

            feature_mat.append(feat_vect1)
            id_list.append(id_label)
            if print_ID:
                print('\n.........case_id {}.....'.format(id_label))
  
        feature_mat=np.vstack(feature_mat)
        print("glcm_matrix {} -dimensional".format(feature_mat.shape))
        
        
        df_mat = pd.DataFrame(feature_mat, columns = feat_names)
        df_id = pd.DataFrame(id_list, columns=list(['ID']))
        df1=pd.concat([df_id,df_mat],axis=1)
        

        self.outname=os.path.join(FileTags.root_path,
                             FileTags.mri_data,
                             FileTags.output_fmatrix,
                             FileTags.time_point, 
                             FileTags.mri_data + '_'+ mri_tag +'_'+ FileTags.name_a + '_pp2.csv')
        
        df1.to_csv(self.outname, index=False)
        print('saved in ...... ', self.outname)
                      
        return df1, feat_names
                      
    def glcm_names(self,FileTags):
        h_names=['glcm_Autocorrelation',
        'glcm_ClusterProminence',
        'glcm_ClusterShade',
        'glcm_ClusterTendency',
        'glcm_Contrast',
        'glcm_Correlation',
        'glcm_DifferenceAverage',
        'glcm_DifferenceEntropy',
        'glcm_DifferenceVariance',
        'glcm_Id',
        'glcm_Idm',
        'glcm_Idmn',
        'glcm_Idn',
        'glcm_Imc1',
        'glcm_Imc2',
        'glcm_InverseVariance',
        'glcm_JointAverage',
        'glcm_JointEnergy',
        'glcm_JointEntropy',
        'glcm_MCC',
        'glcm_MaximumProbability',
        'glcm_SumAverage',
        'glcm_SumEntropy',
        'glcm_SumSquares']
        
        s_names = ['median', 'var', 'skew', 'kurt']
        feat_names = []

        if FileTags.mri_mod in ["T1c","CT1","T1_C"]: # standardizing the MRI sequence tag
            mri_tag="T1c"
        else:
            mri_tag=FileTags.mri_mod

        for ws in self.kernel_radius:
            for bins in self.bines:
                for h in h_names:
                    for s in s_names:
                        feat_names.append(mri_tag+'_'+s + '_' + h + '_' + str(ws)+  '_bin_' + str(bins)+'_'+FileTags.sufx)

        
        print("number of glcm-derived features: {}, 4 statistics : {}".format(len(h_names),s_names))
        print(" number of window sizes: {}, number of bin levels: {}".format(self.kernel_radius,self.bines))
        print("total number of features per case {}".format(len(feat_names)))

                      
        return h_names, feat_names, mri_tag
    
    def multifeature_merge(self,rois,im_features,mri_mod_merge, out_tag, include_biomodel=False, include_all_shape=False, include_clinics=False, write_csv=False):

        self.root_path = self.params['root_path']
        self.params['habitat'] = rois
        self.params['mri_mod'] = mri_mod_merge
        self.mri_data = self.params['mri_sites'][self.params['mri_data']]

        paths_csv=[]
        dfs=[]

        for feat in im_features:
            self.params['feature']=feat
            feature_path=self.get_featuremat_paths(self.mri_data)
            df_temp=pd.read_csv(feature_path, dtype={0: str})
            #df_temp['ID'].astype(str).str.zfill(2)
            print(f" Adding {os.path.basename(feature_path)} features: dim {df_temp.shape}")
            paths_csv.append(feature_path)
            dfs.append(df_temp)

        if include_all_shape:
            self.params['feature']="feat3"
            feature_path=self.get_featuremat_paths(self.mri_data,non_modality=True,All_tumor=True)
            df_temp=pd.read_csv(feature_path, dtype={0: str})
            print(f" Adding {os.path.basename(feature_path)} features: dim {df_temp.shape}")
            dfs.append(df_temp)
            self.params['feature']=feat # to keep the original feature query

        if include_biomodel:
            self.params['feature']="feat6"
            feature_path=self.get_featuremat_paths(self.mri_data,non_modality=True,All_tumor=True)
            df_temp=pd.read_csv(feature_path, dtype={0: str})
            print(f" Adding {os.path.basename(feature_path)} features: dim {df_temp.shape}")
            dfs.append(df_temp)
            self.params['feature']=feat # to keep the original feature query
            #feature_tag=self.params['feature_family'][self.params['feature']]+"Shape"

        if include_clinics:
            self.params['feature']="feat8"
            feature_path=self.get_featuremat_paths(self.mri_data,non_modality=True,include_clinics=True)
            df_temp=pd.read_csv(feature_path, dtype={0: str})
            print(f" Adding {os.path.basename(feature_path)} features: dim {df_temp.shape}")
            dfs.append(df_temp)
            self.params['feature']=feat # to keep the original feature query
            #feature_tag=self.params['feature_family'][self.params['feature']]+"Shape"



        df_merged=reduce(lambda left, right: pd.merge(left, right, on='ID', how='inner'), dfs)
        print(f"\n size shape merged {df_merged.shape}")

        outname_merged=os.path.join(os.path.dirname(feature_path),
                                    self.mri_data + '_' + self.params['mri_mod'] + '_' +
                                    self.params['habitats'][self.params['habitat']]+ '_'+
                                    out_tag+'_'+self.params['mat_str'])
        
        self.outname=outname_merged
        if write_csv:
            df_merged.to_csv(outname_merged, index=False)
            print(f"merged saved in .. {outname_merged}")
        else:
            print("\nEnable multifeature_merge(write_csv=True) to save csv file (deafult : False)")
            print(f"csv file has not been saved .. {outname_merged}")
        return df_merged
    
    def multihabitat_merge(self,rois_to_merge,im_features,mri_mod, out_tag, include_biomodel=False, include_all_shape=False, include_clinics=False, write_csv=False):

        self.root_path = self.params['root_path']
        self.params['mri_mod'] = mri_mod
        self.mri_data = self.params['mri_sites'][self.params['mri_data']]
        self.params['feature']=im_features

        paths_csv=[]
        dfs=[]

        for RoI in rois_to_merge:
            self.params['habitat']=RoI
            feature_path=self.get_featuremat_paths(self.mri_data)
            df_temp=pd.read_csv(feature_path, dtype={0: str})
            #df_temp['ID'].astype(str).str.zfill(2)
            print(f" Adding {os.path.basename(feature_path)} features: dim {df_temp.shape}")
            paths_csv.append(feature_path)
            dfs.append(df_temp)

        feature_tag=self.params['feature_family'][self.params['feature']]

        if include_all_shape:
            self.params['feature']="feat3"
            feature_path=self.get_featuremat_paths(self.mri_data,non_modality=True,All_tumor=True)
            df_temp=pd.read_csv(feature_path, dtype={0: str})
            print(f" Adding {os.path.basename(feature_path)} features: dim {df_temp.shape}")
            dfs.append(df_temp)
            self.params['feature']=im_features # to keep the original feature query
            feature_tag=feature_tag+"Shape"

        if include_biomodel:
            self.params['feature']="feat6"
            feature_path=self.get_featuremat_paths(self.mri_data,non_modality=True,All_tumor=True)
            df_temp=pd.read_csv(feature_path, dtype={0: str})
            print(f" Adding {os.path.basename(feature_path)} features: dim {df_temp.shape}")
            dfs.append(df_temp)
            self.params['feature']=im_features # to keep the original feature query
            feature_tag=feature_tag+"BioM"
            

        if include_clinics:
            self.params['feature']="feat8"
            feature_path=self.get_featuremat_paths(self.mri_data,non_modality=True,include_clinics=True)
            df_temp=pd.read_csv(feature_path, dtype={0: str})
            print(f" Adding {os.path.basename(feature_path)} features: dim {df_temp.shape}")
            dfs.append(df_temp)
            self.params['feature']=im_features # to keep the original feature query
            feature_tag=feature_tag + "DemoClinics"



        df_merged=reduce(lambda left, right: pd.merge(left, right, on='ID', how='inner'), dfs)
        print(f"\n size shape merged {df_merged.shape}")

        outname_merged=os.path.join(os.path.dirname(feature_path),
                                    self.mri_data + '_' + self.params['mri_mod'] + '_' +
                                    out_tag + '_'+ feature_tag + '_' +
                                    self.params['mat_str'])
        
        self.outname=outname_merged
        if write_csv:
            df_merged.to_csv(outname_merged, index=False)
            print(f"merged saved in .. {outname_merged}")
        else:
            print("\nEnable multifeature_merge(write_csv=True) to save csv file (deafult : False)")
            print(f"csv file has not been saved .. {outname_merged}")
        return df_merged
    
    def multisequence_merge(self,rois,im_features,mri_mod_merge, out_tag, include_biomodel=False, include_shape=False, write_csv=False):

        self.root_path = self.params['root_path']
        self.params['habitat'] = rois
        self.params['feature']=im_features
        self.mri_data = self.params['mri_sites'][self.params['mri_data']]

        paths_csv=[]
        dfs=[]

        for mod in mri_mod_merge:
            self.params['mri_mod']=mod
            feature_path=self.get_featuremat_paths(self.mri_data)
            df_temp=pd.read_csv(feature_path, dtype={0: str})
            #df_temp['ID'].astype(str).str.zfill(2)
            print(f" Adding {os.path.basename(feature_path)} features: dim {df_temp.shape}")
            paths_csv.append(feature_path)
            dfs.append(df_temp)

        feature_tag=self.params['feature_family'][self.params['feature']]

        if include_shape:
            self.params['feature']="feat3"
            feature_path=self.get_featuremat_paths(self.mri_data,non_modality=True,All_tumor=True)
            df_temp=pd.read_csv(feature_path, dtype={0: str})
            print(f" Adding {os.path.basename(feature_path)} features: dim {df_temp.shape}")
            dfs.append(df_temp)
            self.params['feature']=im_features # to keep the original feature query
            feature_tag=feature_tag+"Shape"

        if include_biomodel:
            self.params['feature']="feat6"
            feature_path=self.get_featuremat_paths(self.mri_data,non_modality=True,All_tumor=True)
            df_temp=pd.read_csv(feature_path, dtype={0: str})
            print(f" Adding {os.path.basename(feature_path)} features: dim {df_temp.shape}")
            dfs.append(df_temp)
            self.params['feature']=im_features # to keep the original feature query
            feature_tag=feature_tag+"BioM"

        df_merged=reduce(lambda left, right: pd.merge(left, right, on='ID', how='inner'), dfs)
        print(f"\n size shape merged {df_merged.shape}")

        self.params['feature']=im_features
        outname_merged=os.path.join(os.path.dirname(feature_path),
                                    self.mri_data + '_' + out_tag + '_' + 
                                    self.params['habitats'][self.params['habitat']]+ '_' +
                                    feature_tag + '_'+self.params['mat_str'])
        

        
        self.outname=outname_merged
        if write_csv:
            df_merged.to_csv(outname_merged, index=False)
            print(f"merged saved in .. {outname_merged}")
        else:
            print("\nEnable multifeature_merge(write_csv=True) to save csv file (deafult : False)")
            print(f"csv file has not been saved .. {outname_merged}")
        
        return df_merged
    
    def multiSequHabFeat_merge(self,rois,im_features,mri_mod_merge, out_tag, include_biomodel=False, include_shape=False, write_csv=False):

        self.root_path = self.params['root_path']
        self.mri_data = self.params['mri_sites'][self.params['mri_data']]

        paths_csv=[]
        dfs=[]
        for data in range(0,2):
            self.params['mri_mod']=mri_mod_merge[data]
            self.params['habitat'] = rois[data]
            self.params['feature']=im_features[data]
            feature_path=self.get_featuremat_paths(self.mri_data)
            df_temp=pd.read_csv(feature_path, dtype={0: str})
            #df_temp['ID'].astype(str).str.zfill(2)
            print(f" Adding {os.path.basename(feature_path)} features: dim {df_temp.shape}")
            paths_csv.append(feature_path)
            dfs.append(df_temp)

        feature_tag=out_tag

        if include_shape:
            self.params['feature']="feat3"
            feature_path=self.get_featuremat_paths(self.mri_data,non_modality=True,All_tumor=True)
            df_temp=pd.read_csv(feature_path, dtype={0: str})
            print(f" Adding {os.path.basename(feature_path)} features: dim {df_temp.shape}")
            dfs.append(df_temp)
            feature_tag=feature_tag+"Shape"

        if include_biomodel:
            self.params['feature']="feat6"
            feature_path=self.get_featuremat_paths(self.mri_data,non_modality=True,All_tumor=True)
            df_temp=pd.read_csv(feature_path, dtype={0: str})
            print(f" Adding {os.path.basename(feature_path)} features: dim {df_temp.shape}")
            dfs.append(df_temp)
            feature_tag=feature_tag+"BioM"

        df_merged=reduce(lambda left, right: pd.merge(left, right, on='ID', how='inner'), dfs)
        print(f"\n size shape merged {df_merged.shape}")

        
        out_tag_mod=f"{mri_mod_merge[0]}{mri_mod_merge[1]}"

        self.params['feature']=im_features
        outname_merged=os.path.join(os.path.dirname(feature_path),
                                    self.mri_data + '_' + out_tag_mod + '_' + 
                                    self.params['habitats'][self.params['habitat']]+ '_' +
                                    feature_tag + '_'+self.params['mat_str'])
        

        
        self.outname=outname_merged
        if write_csv:
            df_merged.to_csv(outname_merged, index=False)
            print(f"merged saved in .. {outname_merged}")
        else:
            print("\nEnable multifeature_merge(write_csv=True) to save csv file (deafult : False)")
            print(f"csv file has not been saved .. {outname_merged}")
        
        return df_merged

    def get_featuremat_paths(self,site, non_modality=False,All_tumor=False, include_clinics=False):
       
        params=self.params
        im_feature=params['feature_family'][params['feature']]
        print("habitat ",params['habitats'][params['habitat']])

        if im_feature=="shape":
            mri_mod='T1c'
        else:
            mri_mod=self.params['mri_mod']


        feature_path = os.path.join(
            self.params['root_path'], 
            site, 
            params['output_fmatrix'], 
            params['time_point'][params["tp"]],
            f"{site}_{mri_mod}_{params['habitats'][params['habitat']]}_{im_feature}_{params['mat_str']}"
        )
        
        if non_modality:
            print(f">>modality_free feature")
            feature_path=feature_path.replace(f"{mri_mod}_",'')
        else:
            print(f">>{mri_mod} MRI features")

        if All_tumor:
            print(f">>habitat_free feature (All tumor)")
            feature_path=feature_path.replace(f"{params['habitats'][params['habitat']]}",'All_tumor')

        if include_clinics:
            print(f">>Demogrpahics and clinical variables ")
            feature_path=feature_path.replace(f"_{params['mat_str']}",'.csv')
            feature_path=feature_path.replace(f"_{params['habitats'][params['habitat']]}",'')
          

        if not os.path.exists(feature_path):
            raise ValueError(f"Missing or non exiting file: {feature_path}")

        return feature_path


    def annotated_matrix(self, df_mat, FileTags, include_clinics=False, write_csv=False):

        labels_path=os.path.join(FileTags.root_path , FileTags.mri_data , 'Metadata',FileTags.mri_data + '-labels.csv' )
        #print('labels_path ..... ',labels_path)
        df_metadata=pd.read_csv(labels_path, dtype={0: str})

        if include_clinics:
            # preventing duplication
            columns_to_drop = ['sex_code', 'Age', 'MGMT_status_code', 'IDH_status_code']

            # Drop them from either DataFrame (e.g., from df_metadata) if they’re not needed for the merge
            df_metadata = df_metadata.drop(columns=columns_to_drop, errors='ignore')

        # Step 1: Merge the DataFrames based on matching identifiers (DF1['ID'] and DF2['ID1'])
        merged_df = pd.merge(df_mat, df_metadata, left_on='ID', right_on='ID1')
        # Step 2: Create the new DataFrame with desired column order
        df_annotated = merged_df[['ID', 'x1_dead_0_alive','OS_days'] + df_mat.columns[1:].tolist()]
        print('\nsize annotated feature matrix', df_annotated.shape)

        ext="_annots.csv"
        annotated_mat=self.outname.replace('.csv',ext)
        #print("annotated output_file ... ", annotated_mat)
        # Display the final DataFrame
        if write_csv:
            df_annotated.to_csv(annotated_mat, index=False)
            print('saved in ...... ', annotated_mat)
        else:
            print("\nEnable annotated_matrix(write_csv=True) to save csv file (deafult : False)")
            print(f"csv file has not been saved .. {annotated_mat}")

        return df_annotated
    
    def merge_dataframes_on_ID(self, dfs, key='ID'):

        df_merged=reduce(lambda left, right: pd.merge(left, right, on=key, how='inner'), dfs)

        return df_merged
    
def mask_binarization(mask_nib, label_val, habitat):
    
    label = np.zeros((mask_nib.shape))
    
    if habitat != "All_tumor":
        label[mask_nib==label_val] = 1 ### consider only specified label
        label_vol=np.where(label==1)
    else:
        label[mask_nib>0] = 1 ### consider the entire mask
        label_vol=np.where(label==1)
        
    return label_vol
    
def create_pyradiomics_params(bin_width, label, distances, voxel_based=False, kernel_radius=0, masked_kernel=False, init_value=0, voxel_batch=1000):
    """
    Args: GLCM set up
    Returns: a dictionary containing the configuration parameters.
    """
    # Default angles for GLCM computation in 3D (13 unique directions)
    angles = [
        (0, 0, 1), (0, 1, 0), (1, 0, 0), 
        (0, 1, 1), (0, 1, -1), (1, 0, 1), 
        (1, 0, -1), (1, 1, 0), (1, -1, 0), 
        (1, 1, 1), (1, 1, -1), (1, -1, 1), 
        (1, -1, -1)
    ]
    # extractor and GLCM-specific level settings
    params = {
        'normalize': False,
        'binWidth': bin_width,
        'resampledPixelSpacing': None,
        'interpolator': None,
        'verbose': False,
        'additionalInfo': False,
        'label': label,
        'distances': distances,
        'voxelBased': voxel_based,
        'kernelRadius':kernel_radius,
        'maskedKernel': masked_kernel,
        'initValue': init_value,
        'voxelBatch': voxel_batch,
        'glcm': {
            'force2D': False,
            'symmetricGLCM': True,
            'angles': angles
        }
    }
    return params

def compute_haralick_stats(haralick_result,print_stats_list=False):
    glcm_stats = {key: value for key, value in haralick_result.items() if 'glcm' in key.lower()}
    glcm_names=list(glcm_stats.keys())
    glcm_stats=[]
    
    for names in list(glcm_names[:]):
        if print_stats_list:
            print("GLCM Stats name: {}".format(names))
            
        glcm_array = haralick_result[names]
        glcm_array = sitk.GetArrayFromImage(glcm_array)
        glcm_values=glcm_array[glcm_array != 0]
        f_median = np.nanmedian(glcm_values)
        f_var = np.nanvar(glcm_values)
        f_skew = st.skew(glcm_values, nan_policy='omit')
        f_kurt = st.kurtosis(glcm_values, nan_policy='omit')
        stats_list=list([f_median,f_var,f_skew,f_kurt])
        glcm_stats.append(stats_list)

    glcm_stats=np.squeeze(glcm_stats)
    print('glcm_stats.shape',(glcm_stats.shape))
    
    return glcm_stats

def get_bounding_box(mask_array):
    non_zero_coords = np.argwhere(mask_array)
    print('number of non zero voxels in mask all the habitats',len(non_zero_coords))
    min_coords = non_zero_coords.min(axis=0)
    max_coords = non_zero_coords.max(axis=0) + 1  # +1 to include the max index
    return tuple(slice(min_c, max_c) for min_c, max_c in zip(min_coords, max_coords))


def get_cropped_images(mri_sitk,mask, **kwargs):
    # Load the mask image
    label_val=kwargs['label_val']
    mask_array = sitk.GetArrayFromImage(mask)
    mask_array[mask_array != label_val] = 0
    bounding_box = get_bounding_box(mask_array)
    
    print('bounding_box coordinates', bounding_box)
    cropped_mask_array = mask_array[bounding_box]
    print('cropped mask', cropped_mask_array.shape)
    cropped_mask_array = cropped_mask_array.astype(float)
    cropped_mask_array[cropped_mask_array == 0] = np.nan

    image_array = sitk.GetArrayFromImage(mri_sitk)
    cropped_image_array = image_array[bounding_box]
    print('cropped image', cropped_image_array.shape)
  
    # Convert back to SimpleITK images (optional set the header information accordingly)
#     cropped_mask = sitk.GetImageFromArray(cropped_mask_array)
#     cropped_image = sitk.GetImageFromArray(cropped_image_array)
    
    return cropped_image_array, cropped_mask_array

def plot_glcm_map(FileTags,cropped_image_array, cropped_mask_array, result, glcm_stats, **kwargs):
    
    mri_mod=kwargs['mri_mod']
    glcm_dict = {key: value for key, value in result.items() if 'glcm' in key.lower()}
    glcm_names=list(glcm_dict.keys())
    i=-1
    for names in list(glcm_names[:]): # (glcm_names[17:18]) to plot specific haralick's statistics
        i=i+1
        print(names)
        output_image=os.path.join(FileTags.output_name,
                                FileTags.mri_mod+'_'+ names+'_' + 
                                FileTags.name_a + '_ws_'+ str(FileTags.ws) +
                                '_bin_'+ str(FileTags.levels/FileTags.binWidth) +
                                '_' + FileTags.pipeline + '.png')
                                
        glcm_entropy = result[names]
        glcm_entropy = sitk.GetArrayFromImage(glcm_entropy)
        glcm_entropy = glcm_entropy.astype(float)
        
        if cropped_image_array.shape != glcm_entropy.shape:
            print(" GLCM map size", glcm_entropy.shape)
            print("Dimension mismatch: Resizing GLCM map. just for visualization purposes")
            zoom_factors = [n / o for n, o in zip(cropped_image_array.shape, glcm_entropy.shape)]
            glcm_entropy = zoom(glcm_entropy, zoom_factors, order=1)
            output_image=output_image.replace('.png','_rz.png')
        
        print('glcm_map shape', glcm_entropy.shape)
        glcm_entropy[cropped_mask_array != FileTags.label_val] = np.nan
        middle_slice_index = cropped_image_array.shape[0] // 2
        # Get middle slices
        image_middle_slice = cropped_image_array[middle_slice_index]
        mask_middle_slice = cropped_mask_array[middle_slice_index]
        glcm_entropy_slice = glcm_entropy[middle_slice_index]

        fig, ax = plt.subplots()

        im = ax.imshow(image_middle_slice, cmap='gray')
        mask_overlay = ax.imshow(glcm_entropy_slice, cmap='jet', alpha=0.4)  # Adjust alpha for transparency

        median_val=glcm_stats[i,0]
        variance_val=glcm_stats[i,1]
        skewness_val=glcm_stats[i,2]
        kurtosis_val=glcm_stats[i,3]
        
        stats_text = f'Median: {median_val:.3f}\nVariance: {variance_val:.3f}\nSkewness: {skewness_val:.3f}\nKurtosis: {kurtosis_val:.3f}'
#         ax[0].text(0.15, 0.4, stats_text, fontsize=20, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))            
#         ax[0].set_title(names +'\n' + str(ws) +' bin '+ str(levels/binWidth))
        ax.axis('off')  # Turn off axis for the MRI image subplot
        ax.set_title(stats_text,fontsize=18, loc='center')
        
        # Remove axis ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    
        cbar = fig.colorbar(mask_overlay, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label('statistic values')
        plt.subplots_adjust(left=0, right=0.8, top=0.76, bottom=0.02)  # Adjust the top parameter to fit the title
        plt.tight_layout(pad=0)

        plt.savefig(output_image,bbox_inches='tight', pad_inches=0)
        
def log_resources():
    # Log CPU and memory usage
    print("CPU usage:", psutil.cpu_percent(interval=1), "%")
    memory_info = psutil.virtual_memory()
    print("Memory usage:", memory_info.percent, "%")
    print(f"Total memory: {memory_info.total / (1024 ** 3):.2f} GB")
    print(f"Available memory: {memory_info.available / (1024 ** 3):.2f} GB")
    print(f"Used memory: {memory_info.used / (1024 ** 3):.2f} GB")