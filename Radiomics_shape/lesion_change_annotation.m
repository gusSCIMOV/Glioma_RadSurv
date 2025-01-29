%% compute radiomics maps 
clear
clc
close 

mri_base='ESLOV'
im_mod='FLAIR'
%%
% aDD FUNCTIONS TOI PATH
mother_path_functions=['\\onfnas01.uwhis.hosp.wisc.edu\radiology\Groups\IDIAGroup\Codebase\gustavo_script\',.......
    '_Neuroimaging\Radiomics_Features_extraction\Radiomics_texture\Matlab_Feat_extraction_radiomic_scripts'];
addpath(genpath(mother_path_functions))

path_root=['\\onfnas01.uwhis.hosp.wisc.edu\radiology\Groups\IDIAGroup\Data\',.....
     '_Brain\Radiology\_Adult\_EM\',mri_base,.......
     '\preprocessed_cimalab\*\time01\*segmentations'];

path_root2=['\\onfnas01.uwhis.hosp.wisc.edu\radiology\Groups\IDIAGroup\Data\',.....
     '_Brain\Radiology\_Adult\_EM\',mri_base,.......
     '\preprocessed_cimalab\*\time02\*segmentations'];
 
out_path=['\\onfnas01.uwhis.hosp.wisc.edu\radiology\Groups\IDIAGroup\Data\',.....
     '_Brain\Radiology\_Adult\_EM\',mri_base,.....
    '\Metadata\'];

d_imgs = dir(path_root);
d_imgs2 = dir(path_root2);

%%
clc
close all
patients=[];
ids=categorical([])
delta_vol=[]
pair_mask=[]
vol1_ids=[]
vol2_ids=[]

for im=16:length(d_imgs)
clear mask vol
in_im=[d_imgs(im).folder,'\',d_imgs(im).name];
mr_split=strsplit(in_im,filesep);
case_id=mr_split{end-2};
tp=mr_split{end-1};

disp(['calculating for ','case id ...',case_id])
disp(['calculating for ','time_point ...',tp])

mri1=dir(['\\',fullfile(mr_split{1:end-1}),'\*IntStnd.nii.gz']);
mri_vol=niftiread(fullfile(mri1.folder,mri1.name));
path_mask1 = fullfile(in_im,'*reg.nii.gz');
dmask1 = dir(path_mask1);

% %%%%%%%%% time point 02
in_im2=[d_imgs2(im).folder,'\',d_imgs2(im).name];
mr_split=strsplit(in_im2,filesep);
mri2=dir(['\\',fullfile(mr_split{1:end-1}),'\*FLAIR*rz_reg.nii.gz']);
mri_vol2=niftiread(fullfile(mri2.folder,mri2.name));
path_mask2 = fullfile(in_im2,'*reg.nii.gz');
dmask2 = dir(path_mask2);

for m=1:length(dmask1)
clear vol1 vol2 mask_file1 mask_file2

mask_file1=[dmask1(m).folder,'\',dmask1(m).name];  
mask_vol1=niftiread(mask_file1);
vol1=length(find(mask_vol1>0))
vol1_ids=cat(1,vol1_ids,vol1);

lesion_id=strsplit(mask_file1,filesep);
lesion_id=lesion_id{end}(1:10);
ids=cat(1, ids,categorical({strcat(case_id,'_',tp,'_',lesion_id)}));
patients=cat(1,patients,case_id);

mask2_idx=false;
for m2=1:length(dmask2)
    mask_file2=[dmask2(m2).folder,'\',dmask2(m2).name];
    if contains(mask_file2, lesion_id)
        mask2_idx=true; 
        mask_vol2=niftiread(mask_file2);
        vol2=length(find(mask_vol2>0));
        delta_temp=((vol2-vol1)/vol1)*100
        break
    end 
end

if mask2_idx 
    
    mri1.name
    mri2.name
    strcat(case_id,'_',tp,'_',lesion_id)
    
    delta_vol=cat(1,delta_vol,delta_temp);
    vol2_ids=cat(1,vol2_ids,vol2);
    outputfolder='.\';
    onset=0;
    prompt = 'plot Y N  ';
    get_plot=input(prompt,'s');
%     get_plot='N';
    if strcmp(get_plot,'Y')
    plot_2d_ROI(mri_vol,mask_vol1,outputfolder,7,'01',onset)
    plot_2d_ROI(mri_vol2,mask_vol2,outputfolder,7,'02',onset)
    subplot(2,2,1)
    imshow(imread('01_overlay.png'))
    title(['vol1  ', num2str(vol1),'..case id...',case_id])
    
    subplot(2,2,2)
    imshow(imread('01_ROI_lesion.png'))
    title(['lesion id  ',lesion_id ])
    
    subplot(2,2,3)
    imshow(imread('02_overlay.png'))
    title(['vol2  ', num2str(vol2), '...change..', num2str(delta_temp)])
    set(gcf, 'WindowState','Maximized')
    
    subplot(2,2,4)
    imshow(imread('02_ROI_lesion.png'))
    title(['vol2  ', dmask2(m2).name])
    pause
     close all
    end
   
else
    delta_vol=cat(1,delta_vol,-100);
    vol2_ids=cat(1,vol2_ids,0);
end   
    
end


end
%%
% T = array2table(feature_mat, 'VariableNames', vect_ws_string);
% T_ids=table(ids,'VariableNames', {'ID'});
% writetable([T_ids,T],[out_path,feature,'_',tm_habitat,'.xlsx'] , 'Sheet','FeatureMatrix');

table(patients,ids,delta_vol,vol1_ids,vol2_ids)
writetable(ans,'HUN_labels.xlsx')
%% split clases by a threshold
clc
mri_base='MSSEG2'
im_mod='FLAIR'
root_path=['\\onfnas01.uwhis.hosp.wisc.edu\radiology\Groups\IDIAGroup\Data\',.....
     '_Brain\Radiology\_Adult\_EM\',mri_base,.......
     '\Metadata\'];
file_path=[root_path,mri_base,'_labels.xlsx'];
data=readtable(file_path);

shrink_t=-100;
 
 for rows=1:size(data,1)
     
   if isempty(data.true_label{rows})
       if data.delta_vol(rows)<=shrink_t
          data.true_label{rows}='shrinking'; 
       else
          data.true_label{rows}='static';
       end
   end
       
 end
 
outname=[root_path,mri_base,'_shrink_',num2str(shrink_t),'_labels.xlsx']
writetable(data,outname)

