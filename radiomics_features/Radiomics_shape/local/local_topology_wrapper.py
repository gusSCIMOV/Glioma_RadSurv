#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 10:10:58 2020

@author: hyeminum
"""

import logging
import topology_radiomics as toprad
import nibabel as nib
import os
import numpy as np
import pandas as pd
FORMAT = '%(asctime)-15s %(levelname)s %(funcName)s  %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)

label_val = 3

# path_to_image = '/Users/hyeminum/Documents/Research/Mets/analyze/training/deform'
path_to_image = '/Volumes/Pebble/Medulloblastoma/CHOP/Radiology/patients'
label_name = "All-label_transform_updated.nii.gz"

labels = []
for (dirpath, dirlist, filelist) in os.walk(path_to_image):
    for filename in filelist:
        if filename.startswith(label_name):
            labels.append(os.path.join(dirpath, filename))
            
labels = sorted(labels)

ids = []
ls_stats = []
for i in labels:
    nib_image = nib.load(i)
    mri_image = nib_image.get_fdata()
    
    path_id = os.path.dirname(i) ### always double check this line 
    pp = [pos for pos, char in enumerate(path_id) if char == '/']
    id_label = path_id[pp[-1]+1:]
    
    print(id_label)
    
    # if label_val in mri_image:
    if label_val < 3 and label_val in mri_image:
        mri_image_habitat = np.zeros((mri_image.shape))
        # if label_val > 0:
        #     mri_image_habitat[mri_image==label_val] = 1 ### consider only specified label
        if 0 < label_val < 3:
            mri_image_habitat[mri_image==label_val] = 1 ### consider only specified label
        # else:
        elif label_val == 0:
            mri_image_habitat[mri_image>0] = 1 ### consider all labels
    elif label_val == 3 and label_val in mri_image or 4 in mri_image:
        mri_image_habitat = np.zeros((mri_image.shape))
        mri_image_habitat[mri_image>=label_val] = 1
    
    # mri_image_habitat = np.zeros((mri_image.shape))
    # if 0 < label_val < 3:
    #     if label_val in mri_image:
    #         mri_image_habitat[mri_image==label_val] = 1
    # elif label_val == 3:
    #     if label_val in mri_image or 4 in mri_image:
    #         mri_image_habitat[mri_image>=label_val] = 1
    # elif label_val == 0:
    #     if label_val in mri_image:
    #         mri_image_habitat[mri_image>0] = 1
        
        print('\n')
        print('\nCalculating Toplogy for:', id_label)
        print('\n')
    
    # if label_val in mri_image:
    # if 1 in mri_image_habitat:
        # merge_labels=['1 for enhancing', '2 for edema', '3 for necrosis']
        # sanitized_mask = toprad.convert_volume_into_mask(mri_image, merge_labels=[label_val])
        sanitized_mask = toprad.convert_volume_into_mask(mri_image_habitat, merge_labels=[1])
        features = toprad.compute_morphology_features(sanitized_mask)
        df = features.to_DF()
        
        # to compute single statistic from data frame
        # mean = df["curvedness"].mean()
        # to compute multiple statistics from data frame
        stats = df.agg({'curvedness': ['mean', 'median', 'var', 'skew', 'kurt'],
           'sharpness': ['mean', 'median', 'var', 'skew', 'kurt'],
           'shape_index': ['mean', 'median', 'var', 'skew', 'kurt'],
           'total_curvature': ['mean', 'median', 'var', 'skew', 'kurt']})
        vals = stats.values
        matT = vals.T
        vect = matT.reshape(1,20)
        ls_stats.append(vect)
        
        # extract patient ID from path to label
        # ids.append(int(''.join(list(filter(str.isdigit, i))[0:4])))
        ids.append(id_label)
        
    else:
        print('\nToplogy not calculated. Label %d not found in mask for:' %label_val, id_label)
        print('\n')
        continue
    
np_stats = np.squeeze(ls_stats)
    
cl = stats.columns.values.tolist()
rw = stats.index.values.tolist()
names = []
for c in cl:
    for r in rw:
        names.append(r + '_' + c)
        
df_stats = pd.DataFrame(np_stats, index = ids, columns = names)

if label_val == 1:
    csvname = '/Topology_Enhancing.csv'
elif label_val == 2:
    csvname = '/Topology_Edema.csv'
elif label_val == 3:
    # csvname = '/Topology_Necrosis.csv'
    csvname = '/Topology_Cyst+NET.csv'
else:
    csvname = '/Topology_Habitat.csv'
        
df_stats.to_csv(path_to_image + csvname, index_label='ID')

