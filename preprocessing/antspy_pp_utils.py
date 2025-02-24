import os
import glob
import subprocess
import numpy as np
import nilearn
import nibabel as nb
import pandas as pd
import ants
import logging
import matplotlib.pyplot as plt

from pathlib import Path
from collections import OrderedDict
from ants import resample_image
from reorient_nii import get_orientation
from reorient_nii import load
from reorient_nii import reorient
from nilearn import plotting
from nilearn.image import resample_to_img
from ants import get_ants_data, image_read, resample_image, get_mask
from pathlib import Path
from tempfile import mkstemp
from utils.config_loader import *
from utils.generic import *

class MRIPreprocessing:
    # intiate and run sequential-step by step preprocessing pipelines in preprocessing_config.yaml
    def __init__(self,dataset, dirs, gen_config, pp_setting, pp_config):
        self.dataset= dataset
        self.input_dir=os.path.join(gen_config.root_path,dataset,dirs.preprocessed)
        self.dirs=dirs
        self.gen_config=gen_config
        self.pp_config=pp_config
        self.pp_settings=pp_setting
        
        self.logger = logging.getLogger(self.__class__.__name__) 

    
    def run_pipeline(self):


        for stepp, params in vars(self.pp_config).items():
            self.params=Config2Struct(params)
            self.logger.info(f" PREPROCESSING {stepp} >>> {self.params.name}") # check config.preprocessing_config.yaml order is correct

            if self.params.name == "Modality_2_atlas_reg":
                
                # set reference imgage (atlas)
                atlas_path= os.path.join(self.gen_config.project_path,'preprocessing/ATLAS_T1',
                                        self.params.atlas_str,'templates')

                atlas_path += '/T1_brain.nii' if self.params.brain_mask is not None else '/T1.nii'
                
                self.mri_ref=self.params.mri_str # save the reference MRI modality
                self.imgs, labels = getting_image_list(self.input_dir,
                                                  self.pp_settings.acquisition_tag, 
                                                  self.mri_ref,
                                                  self.params.ext,
                                                  self.params.mask_str)
                self.logger.debug('number of MRI scans: {} , for MRI modality: {}'.format(len(self.imgs), self.params.mri_str))

                self.atlas_regis_ls=[]
                for im in self.imgs:
                    registered_path, id_label=mri_coregistration(im,self.params.ext,
                                        self.params.type_of_transform,
                                        self.params.aff_metric,
                                        self.dirs.preprocessed, #input subdir
                                        self.dirs.preprocessed, #output subdir
                                        atlas_path=atlas_path,
                                        atlas_str=self.params.atlas_str,
                                        mapping="forward_mapping", 
                                        write_registered=True, ID_level=-3)
                    self.atlas_regis_ls.append((im,registered_path))
                
            if self.params.name == "Modalities_coregistration":
                
                mri_ls= list(filter(lambda x: x != self.mri_ref, self.pp_settings.mri_modalities)) # filtering ref mri (first step)
                coregist_ls=[]

                for im , reference_mri in self.atlas_regis_ls:
                    #print("debug ....",im, reference_mri)        
                    mods_coregist={} # initialize log dict
                    for mri_mod in mri_ls:  
                        mods_coregist.update({mri_mod: "NO_Modality"})
                    
                    id_label=Path(im).parts[-3] # Should be included in config_preprocessing.yaml

                    for mri_mod in mri_ls:
                        mov_im=im.replace(self.mri_ref,mri_mod)
                        
                        if os.path.exists(mov_im):
                            registered_path, _ = mri_coregistration(mov_im, self.params.ext,
                                                self.params.type_of_transform,
                                                self.params.aff_metric,
                                                self.dirs.preprocessed, #input subdir
                                                self.dirs.preprocessed, #output subdir
                                                template=reference_mri,
                                                mapping="forward_mapping", 
                                                write_registered=True, ID_level=-3) 
                            
                            mods_coregist.update({mri_mod: registered_path})
                            
                        else:
                            continue

                    mods_coregist = OrderedDict([('ID', id_label)] + 
                                                [(self.mri_ref, reference_mri)] +
                                                list(mods_coregist.items()))
                           
                    coregist_ls.append(mods_coregist)

                self.coregist_df=pd.DataFrame(coregist_ls)
                self.out_df=os.path.join(self.gen_config.root_path,self.dataset,self.dirs.metadata)
                os.makedirs(self.out_df, exist_ok=True)
                outfile=f"{self.out_df}/{self.dataset}_{self.params.name}.csv"
                self.coregist_df.to_csv(outfile, index=False)
            
            if self.params.name == "HD_SkullStripp":

                logging.debug("HD-BET Skull Stripping, fix around to enable alimited number of GPUs")
                self.skulls_df = self.coregist_df.copy()
                ssimages=[]
                # reference image    
                for ind,im in enumerate(self.coregist_df.iloc[:, 1].tolist()):
                    try:
                        masked_im, brain_mask=SkullStrip_HD_Bet(im, cuda_device="cuda")
                        ssimages.append((masked_im,brain_mask))
                        self.skulls_df.iloc[ind,1]=masked_im
                        print("masked_im >>>>>", masked_im)

                            
                    except Exception as e:
                        print(f"\n any exception ocurred {e}")
                        continue 

                # remaining modalities
                for ind, (_, brain_mask) in enumerate(ssimages): 
                    for col in self.coregist_df.columns[2:]:  # I
                        mri_mod = self.coregist_df.loc[ind, col]

                        if mri_mod != "NO_Modality":
                            masked_im=SkullStripp_WithMask(mri_mod, brain_mask)
                            self.skulls_df.loc[ind,col]=masked_im
            
                os.makedirs(self.out_df, exist_ok=True)
                outfile=f"{self.out_df}/{self.dataset}_{self.params.name}.csv"
                self.skulls_df.to_csv(outfile, index=False)


def mri_coregistration(im,ext,type_transform,aff_metric,subdir,out_subdir,
                        template=None, atlas_path=None,atlas_str=None,neg_mask=None,
                        im_seg=None, brainmask=None, mapping="forward_mapping", write_registered=True, ID_level=-3):
    #ext=".nii.gz"
    ext_prealign = "_align"+ ext

    # Set the reference image (atlas)
    if not atlas_path == None:
        atlas=atlas_path
        ext_reg = "_"+ atlas_str + ".nii.gz"
        print(f"\n >>>>>>> ... mri_coregistration towards {atlas_str} ")
    else:
        atlas=template
        ext_reg = "_reg.nii.gz"
        print(f"\n >>>>>>> ... mri_coregistration towards {os.path.basename(template)} ")

    reference_im = ants.image_read(atlas)

    #  >>>> LOGS 
    id_label=Path(im).parts[ID_level]
    print(f"\n case ID {id_label}  moving image {os.path.basename(im)} reference {os.path.basename(atlas)}" )

    #  >>>> MRI Pre-aligment  / output paths

    moving_t1 = get_info_match_tp1(atlas,im,ext,ext_prealign) 
    print('Prealigned.....',moving_t1) 
    moving_im = ants.image_read(moving_t1)
    #os.makedirs(os.path.dirname(omoving_t1), exist_ok=True)
    omoving_t1 = moving_t1.replace(ext_prealign,ext_reg)
    omoving_t1 = omoving_t1.replace(subdir, out_subdir)
    
    if os.path.exists(omoving_t1):
        print(f" {omoving_t1} already exists")
        os.remove(moving_t1)
        return omoving_t1, id_label

    #  >>>> Segmentations (RoI and Brain mask) Pre-aligment  / output paths

    if im_seg != None:
        #print('moving mask........',os.path.basename(im_seg))
        moving_segr = get_info_match_tp1_seg(atlas,im_seg,ext,ext_prealign) # check mask extension inside this function
        seg_im = ants.image_read(moving_segr)
        omoving_segr = moving_segr.replace(ext_prealign, ext_reg)
        omoving_segr=omoving_segr.replace(subdir, out_subdir)
    else:
        seg_im= None

    if brainmask !=None:
        #print('moving brain mask........',os.path.basename(brainmask))
        moving_brainmask = get_info_match_tp1_seg(atlas,brainmask,ext,ext_prealign) # check mask extension inside this function
        brainmask_im = ants.image_read(moving_brainmask)
        output_brainmask = moving_brainmask.replace(ext_prealign, ext_reg)
        output_brainmask=output_brainmask.replace(subdir, out_subdir)
    else:
        brainmask_im= None

    # set masked region for resgitration ()
    if neg_mask !=None:
        non_tumor_mask=neg_mask
        moving_non_tumor_mask = get_info_match_tp1_seg(atlas,non_tumor_mask,ext,ext_prealign)
        non_tumor_mask_im=ants.image_read(moving_non_tumor_mask)
    else:
        non_tumor_mask_im=None
        
    # Run MRI registration

    if mapping=="forward_mapping":
        print(f"Fitting {type_transform} forward_mapping")
        warpedimage, warpedseg, warped_brainmask = run_forward_coregistration(
                                        fixed_im=reference_im,
                                        moving_im=moving_im, 
                                        transform=type_transform, 
                                        regis_metric=aff_metric, 
                                        masked=non_tumor_mask_im, 
                                        RoI_mask=seg_im, 
                                        brain_mask=brainmask_im)  

    elif mapping=="InverseWarp_mapping":
        print(f"Fitting {type_transform} InverseWrap_mapping")
        warpedimage, warpedseg, warped_brainmask = run_InverseWarp_registration(
                                        fixed_im=moving_im,
                                        moving_im=reference_im,
                                        transform=type_transform, 
                                        regis_metric=aff_metric,
                                        masked=non_tumor_mask_im, 
                                        RoI_mask=seg_im,
                                        brain_mask=brainmask_im)



    # >>>> writting image ouput amd temp files (pre-aligned)
    if write_registered:
        ants.image_write(warpedimage, omoving_t1)
        print(f"registred image saved in : {omoving_t1}")
    
    os.remove(moving_t1)

    if im_seg != None:
        # writting Seg_mask 
        ants.image_write(warpedseg, omoving_segr)
        os.remove(moving_segr)
    
    if neg_mask !=None:
        os.remove(moving_non_tumor_mask)

    if brainmask !=None:
        # writting Seg_mask 
        ants.image_write(warped_brainmask, output_brainmask)
        os.remove(moving_brainmask)

    print(f"\n{mapping} completed for {id_label}")

    return omoving_t1, id_label


def run_forward_coregistration(fixed_im=None,moving_im=None,
                                transform=None, 
                                regis_metric=None,masked=None, 
                                RoI_mask=None,brain_mask=None):
    
    # fit transformation
    mytx = ants.registration(
        fixed=fixed_im,
        moving=moving_im,
        mask=masked,
        type_of_transform=transform,
        aff_metric=regis_metric)

    # Apply image forward mapping
    warpedimage = ants.apply_transforms(
        fixed=fixed_im, 
        moving=moving_im, 
        transformlist=mytx['fwdtransforms'])
 
    if RoI_mask != None:
        warpedimage_seg = ants.apply_transforms(
            fixed=fixed_im, 
            moving=RoI_mask, 
            interpolator='nearestNeighbor',
            transformlist=mytx['fwdtransforms'])
    else:
        warpedimage_seg=None
    
    if brain_mask !=None:
        warped_brainmask = ants.apply_transforms(
            fixed=fixed_im,
            moving=brain_mask,
            transformlist=mytx['fwdtransforms'],
            interpolator='nearestNeighbor')
    else:
        warped_brainmask=None
      
    return warpedimage, warpedimage_seg, warped_brainmask

def run_InverseWarp_registration(fixed_im=None,moving_im=None,
                                    transform=None, 
                                    regis_metric=None,masked=None, 
                                    RoI_mask=None,brain_mask=None):
    
    # fit transformation
    mytx = ants.registration(
        fixed=fixed_im,
        moving=moving_im,
        type_of_transform=transform,
        mask=masked,
        reg_iterations=(100, 100, 100, 20),
        grad_step=0.25,
        aff_metric=regis_metric,  # Mattes MI in ANTsPy, similar to MI
        aff_sampling=32,  # Matching bin sampling
        syn_metric=regis_metric,
        syn_sampling=32,
        syn_sigma=(3, 0),
        verbose=False
    )

    warpedimage = ants.apply_transforms(
        fixed=moving_im,
        moving=fixed_im,
        transformlist=mytx['invtransforms'],
        interpolator='linear'
    )

    if RoI_mask != None:
        warpedseg = ants.apply_transforms(
            fixed=moving_im,
            moving=RoI_mask,
            transformlist=mytx['invtransforms'],
            interpolator='nearestNeighbor'
        )
    else: 
        warpedseg= None

    if brain_mask !=None:
        warped_brainmask = ants.apply_transforms(
            fixed=moving_im,
            moving=brain_mask,
            transformlist=mytx['invtransforms'],
            interpolator='nearestNeighbor'
        )
    else:
        warped_brainmask=None

    return warpedimage, warpedseg, warped_brainmask

def getting_neg_mask(seg_mask, label_range=[1,5], save_binary=False):
    
    lesion_path = seg_mask  # Replace with actual lesion path
    output_path = os.path.dirname(lesion_path)  # Replace with actual output path

    # Load the lesion image
    lesion_image = ants.image_read(lesion_path)

    # 1. Replace all lesion labels with 1
    lesion_array = lesion_image.numpy()
    lesion_array[(lesion_array >= label_range[0]) & (lesion_array <= label_range[1])] = 1

    # Convert back to ANTsImage
    lesion_binary = ants.from_numpy(lesion_array, origin=lesion_image.origin,
                                    spacing=lesion_image.spacing, direction=lesion_image.direction)

    if save_binary:
        # Save the modified lesion binary image
        all_label_binary_path = f"{output_path}/ALL-label_binary.nii.gz"
        if os.path.exists(all_label_binary_path):
            os.remove(all_label_binary_path)
        ants.image_write(lesion_binary, all_label_binary_path)
        print(f"\nSaved lesion binary image to {all_label_binary_path}")

    # 2. Generate the negative lesion mask with 1 in the background and 0 in lesion area
    neg_lesion_array = np.ones(lesion_array.shape, dtype=lesion_array.dtype)
    neg_lesion_array[lesion_array == 1] = 0  # Set lesion areas to 0

    # Convert back to ANTsImage
    neg_lesion_image = ants.from_numpy(neg_lesion_array, origin=lesion_image.origin, 
                                       spacing=lesion_image.spacing, direction=lesion_image.direction)

    # Save the negative lesion image
    neg_lesion_path = f"{output_path}/Neg_Lesion.nii.gz"
    if os.path.exists(neg_lesion_path):
        os.remove(neg_lesion_path)
    ants.image_write(neg_lesion_image, neg_lesion_path)
    print(f"\nSaved negative lesion image to {neg_lesion_path}")

    return neg_lesion_path

def BrainmaskBinarization(skullstripped_im):
    # for images having True zero-valued background voxels
    outname=skullstripped_im.replace('.nii.gz','_BrainMask.nii.gz')
    img = nb.load(skullstripped_im)
    data = img.get_fdata()
    binary_mask = np.zeros(data.shape, dtype=np.int16)
    binary_mask[data > 0.1] = 1
    binary_img = nb.Nifti1Image(binary_mask, img.affine, img.header)
    nb.save(binary_img, outname)

    print(f"Binary mask saved as {outname}")
    
    return outname

def ResampToTemplate(img,ref_img):
    print("Image resampling using  one imge as template")
    mov_img = ants.image_read(img)
    temp_im= ants.image_read(ref_img)
    resampled_im = ants.resample_image_to_target(mov_img, temp_im, interp_type=0)
    resampled_name=img.replace(".nii.gz", "_rz.nii.gz")
    ants.image_write(resampled_im, resampled_name)
    print('saving resampled......',resampled_name)

    return resampled_name

def SkullStripp(im, patient_mask=False, sk_method ='HD_BET', atlas_path=None):

    if not patient_mask:
        if sk_method=='atlas_mask':
                    brain_mask_path=os.path.join(atlas_path,'T1_brain_mask.nii.gz')
                    masked_im_path=SkullStripp_WithMask(im, brain_mask_path)
                        
        elif sk_method=='HD_BET':
                    masked_im_path=SkullStrip_HD_Bet(im, cuda_device=0)
    else:

        masked_im_path=SkullStripp_WithMask(im, brain_mask_path)

    return masked_im_path


def SkullStripp_WithMask(image_path, brain_mask_path):
    print('Skullstripping with Mask ..................................')
    masked_im_path=image_path.replace(".nii.gz", "_SkullS.nii.gz")  
    image = ants.image_read(image_path)
    mask = ants.image_read(brain_mask_path)

    if not mask.unique().tolist() == [0, 1]:
        raise ValueError("Mask image is not binary. Please ensure the mask image contains only 0 and 1 values.")

    product_mask = image * mask
    
    product_mask.set_spacing(image.spacing)
    product_mask.set_origin(image.origin)
    product_mask.set_direction(image.direction)

    ants.image_write(product_mask,masked_im_path)
    print('\n SkullStrip done .......',os.path.basename(masked_im_path))
    
    return masked_im_path

# def SkullStrip_HD_Bet(im,cuda_device=0):
    
#     out_path=im.replace('.nii.gz','_SkullS')
#     print("Running SkullStripp {}, HD-BET on cuda device {}".format(os.path.basename(im),cuda_device))
    
#     !hd-bet -i $im -o $out_path -device $cuda_device
#     masked_im_path=out_path + ".nii.gz"
#     return masked_im_path

def SkullStrip_HD_Bet(im, cuda_device=0):
    
    out_path = im.replace('.nii.gz', '_SkullS.nii.gz')
    masked_im_path = out_path.replace('.nii.gz', "_bet.nii.gz")

    #print("Running SkullStripp {}, HD-BET on cuda device {}".format(os.path.basename(im), cuda_device))
    if not os.path.exists(out_path):
        
        # Run HD-BET command using subprocess
        try:
            subprocess.run([
                'hd-bet',
                '-i', im,
                '-o', out_path,
                '-device', str(cuda_device),
                '--save_bet_mask'
            ], check=True)
        except subprocess.CalledProcessError as e:
            print("An error occurred while running HD-BET:", e)
            return None
    else:
        print(f"{out_path} already exists")

    return out_path, masked_im_path

def N4Biasfield_correction(image_path):
        
    o_biascorrect=image_path.replace(".nii.gz", "_BiasCorrect.nii.gz")
    if os.path.exists(o_biascorrect):
        print(f"{o_biascorrect} already exists")
        return o_biascorrect
    
    image = ants.image_read(image_path)

    # Parametrization
    corrected_image = ants.n4_bias_field_correction(
        image, 
        shrink_factor=2, 
        convergence={'iters': [20, 20, 20, 40], 'tol': 0.0005}
    )
    # Save the corrected image
    ants.image_write(corrected_image, o_biascorrect)
    print('\n >>>>> N4 Bias Correction done .......',os.path.basename(o_biascorrect))
    
    return o_biascorrect

def intensity_norm(image_path, mri_mod, pipeline, mask_str=None, Imin=0.0, Imax=1.0, rescaling_method='z-score',ID_level=-3):
    """
    Reads a Brain MRI NIfTI file and its brain mask, handles int16 data type,
    and performs min-max re-scaling on the intensities within the brain region.
    """
    results_range=[]
    # Create a 2-row, 1-column subplot
    fig, axs = plt.subplots(2, 1, figsize=(15, 10))
    
    for imgs in list(image_path[:]):
        
        id_label = Path(imgs).parts[ID_level]
        time_point = Path(imgs).parts[ID_level+1]
        print('\n Patient ID: {}, time point:{}'.format(id_label,time_point))
        print(id_label + '_' + mri_mod)
        print(imgs)
        
        output_image_path = imgs.replace('.nii.gz', '_IntStnd.nii.gz')
       
        img = nb.load(imgs)
        img_header = img.header
        img_affine = img.affine
        image_orig = img.get_fdata()
        
        print(f"Original image data type: {img_header.get_data_dtype()}")
        print(f"Minimum value in the original image: {image_orig.min()}")
        print(f"Maximum value in the original image: {image_orig.max()}")    
        
        if not mask_str==None:
            brain_region, mask_orig=masked_region(imgs, image_orig, mask_str, pipeline)
            brain_region_rescaled=normalizing_intensities(brain_region,rescaling_method,Imax)
            rescaled_data = np.copy(image_orig)
            rescaled_data[mask_orig] = brain_region_rescaled
            rescaled_data[~mask_orig] = 0
            
            processed_image = rescaled_data
            print('type processed_image', type(processed_image))
            print(f"Processed image data type: {processed_image.dtype}")
            print(f"Minimum value in the processed image: {processed_image[mask_orig].min()}")
            print(f"Maximum value in the processed image: {processed_image[mask_orig].max()}")

            
        else:
            print("Intensity normalization without masking")
            brain_region = image_orig
            brain_region_rescaled=normalizing_intensities(brain_region,rescaling_method,Imax)
            processed_image = brain_region_rescaled
            print(f"Minimum value in the processed image: {processed_image.min()}")
            print(f"Maximum value in the processed image: {processed_image.max()}")
            
        #rescaled_data=rescaled_data.astype(np.float32)       
           
        new_header = img.header.copy()
        new_header.set_data_dtype(np.float32)

        # Create the new Nifti image with the processed data, using the original affine and header
        processed_img_nii = nb.Nifti1Image(processed_image, img_affine, new_header)

        
        
        # TO DONT OVERWRITE EXISTING FILES by nib.either nii or nii.gz with the same name
        if os.path.isfile(output_image_path):
            print('existing file removing')
            os.remove(output_image_path)
            
        nb.save(processed_img_nii, output_image_path)
        print("saved in ......", output_image_path)
    
        if not mask_str==None:
            
            axs[0].hist(image_orig[mask_orig].ravel(), bins=256, range=(image_orig[mask_orig].min(), image_orig[mask_orig].max()), histtype='step', label='Original Image')
            axs[1].hist(processed_image[mask_orig].ravel(), bins=256, range=(processed_image[mask_orig].min(), processed_image[mask_orig].max()), histtype='step', label='processed_image')
            results_range.append([processed_image[mask_orig].min(),processed_image[mask_orig].max()])

        else:
            axs[0].hist(image_orig.ravel(), bins=256, range=(image_orig.min(), image_orig.max()), histtype='step', label='Original Image')
            axs[1].hist(processed_image.ravel(), bins=256, range=(processed_image.min(), processed_image.max()), histtype='step', label='processed_image')
            results_range.append([processed_image.min(),processed_image.max()])
        
        
    axs[0].set_title('Overlayed Histograms of First 5 Iterations')
    axs[0].set_xlabel('gray levels')
    axs[0].set_ylabel('log counts')
    axs[0].set_yscale('log')
    axs[0].grid(which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    
    axs[1].set_xlabel('gray levels')
    axs[1].set_ylabel('log counts')
    axs[1].set_yscale('log')
    axs[1].grid(which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    output_image=str(Path(imgs).parents[2])+'/UCSF-PDGM_'+ mri_mod +'_IntStnd_py.png'
    plt.savefig(output_image,bbox_inches='tight', pad_inches=0)    
    
    return results_range

def masked_region(img, image_orig, mask_str, pipeline, t1w_str="T1c"):
    # Pairing Label mask file with the MRI scan
    
    mask_path=img.replace(pipeline,mask_str)
    mask_path=mask_path.replace(t1w_str ,"T1c")
    print(f"brain mask : {mask_path}")
    
    if not os.path.exists(mask_path):
        print(f"Error: File brain mask does not exist.")
        sys.exit(1)  # Exit the program with a non-zero exit code
    else:
        print("Brain mask ....",os.path.basename(mask_path))
    
    mask = nb.load(mask_path)
    mask_orig =  mask.get_fdata().astype(np.bool_)
    
    brain_region = image_orig[mask_orig]
    
    return brain_region, mask_orig

def normalizing_intensities(brain_region,rescaling_method,Imax=None):
    
    if rescaling_method == 'linear_min_max':
            
        brain_min = brain_region.min()
        print('brain_min',brain_min)
        brain_max = brain_region.max()
        print('brain_max',brain_max)
        bw =  (brain_region.max() - brain_region.min()) / Imax
        brain_region_rescaled = (brain_region - brain_region.min()) / bw
        
    elif rescaling_method == 'z-score':
        print("brain_region ... ", type(brain_region))
        brain_region_rescaled= (brain_region-np.mean(brain_region))/np.std(brain_region)
        
        print("mean region with standardization",np.mean(brain_region_rescaled))
        print("std region with standardization",np.std(brain_region_rescaled))
    
    return brain_region_rescaled


def get_files(folder_path,subject_folder, prefix):

    files = folder_path + '/' +  prefix + '_' +subject_folder +  '.nii.gz' 
    files = files.replace('\\', '/')
    
    return files

def get_all_seg_files(folder_path,subject_folder,tp1,prefix):
            
    files = folder_path + '/' + subject_folder + '_' + tp1 + '_' + prefix 
    files = files.replace('\\', '/')
    
    return files

def match_origins(ref_img,img):
    
    ref_aff = ref_img.affine
    img_aff = img.affine
    img_aff[:3,3]=ref_aff[:3,3]
    
    return img
    
def get_info_match_tp1(ref_path,img_path,ext,ext_prealign):

    ref_img= nb.load(ref_path)
#     print("ref_img",ref_path,"info",ants.image_read(ref_path))
    
    img = nb.load(img_path)
#     print("img",img_path, "info",ants.image_read(img_path))
    
#     img_r = resample_to_img(img, ref_img,interpolation='nearest')
#     img_roo = img_r
#     print("img_r",img_r, "info",ants.image_read(img_r))

#     img_ro = reorient(img_r, orientation="LPS")
#     print("img_ro",img_ro)
    
    img_roo = match_origins(ref_img,img)

    path_r = img_path.replace(ext,ext_prealign)
    nb.save(img_roo,path_r)
#     print("img_reg",path_r, "info",ants.image_read(path_r))

    return path_r

def get_info_match_tp1_seg(ref_path,img_path,ext,ext_prealign):
    
    ref_img= nb.load(ref_path)

    img = nb.load(img_path)
#     print("img",img_path, "info",ants.image_read(img_path))
    
#     img_r = resample_to_img(img, ref_img,interpolation='nearest')
#     img_ro = reorient(img_r, orientation="LPS") 
    img_roo = match_origins(ref_img,img)

    path_r= img_path.replace(ext,ext_prealign)
    nb.save(img_roo,path_r)
#     print("img_reg",path_r, "info",ants.image_read(path_r))

    return path_r

def bin_mask(img_path,out_path):
    
    mask = nb.load(img_path)
    mask_data = mask.get_fdata()
    maskdata_bin = np.where(mask_data > 0, 1, 0)
    mask_bin = nb.Nifti1Image(maskdata_bin, mask.affine)
    nb.save(mask_bin, out_path)
    
    return

