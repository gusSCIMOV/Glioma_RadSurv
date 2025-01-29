%% intensiity standardization with linear scaling xCURES

% Intensity standardization
% Hyemin Um, May 2022

% Dependencies:
% int_stdn_landmarks_N.m
% calcstdnmap.m
% applystdnmap.m
% botclip.m
% rescale_range.m
% topclip.m
clear
clc
close 
% aDD FUNCTIONS TOI PATH
mother_path='C:\Users\GXP013\Documents\Medical_imaging\MRI\Scripts_functions';
addpath(genpath(mother_path))
im_mod='T1_C'
t_p='01'
% Generate a template from N randomly chosen subjects
%
% XCURES
% path_root=['C:\Users\GXP013\Documents\Medical_imaging\MRI\GBM_xCures\Preprocessing\Total_pp_results\Nifti\'];
% path_root_mask=['C:\Users\GXP013\Documents\Medical_imaging\MRI\GBM_xCures\Preprocessing\Total_pp_results\Nifti\'];

% LUMIERE (seg 3 >>enhacing tumor,  2 >>necrosis, 1>>> edema)
%path_root=['\\onfnas01.uwhis.hosp.wisc.edu\radiology\Groups\IDIAGroup\Stephen_Transfers\LUMIERE\'];
path_root_mask=['C:\Users\GXP013\Documents\Medical_imaging\MRI\GBM_xCures\Radiomics\deformation_features\Nifti\'];
addpath(genpath(path_root_mask))
patients=dir(path_root_mask)

%%
clc
addpath(['C:\Users\GXP013\Documents\Medical_imaging\MRI\GBM_xCures\meta_data\'])
annots=readtable(['C:\Users\GXP013\Documents\Medical_imaging\MRI\GBM_xCures\meta_data\QC_Xcures.xlsx'])
annots.patient_ID=categorical(annots.patient_ID)
%annots.timepoint=categorical(annots.timepoint)
annots.Pre_operativeScans=categorical(annots.Pre_operativeScans)
annots.Diagnosis=categorical(annots.Diagnosis)
 
%

labels=readtable(['C:\Users\GXP013\Documents\Medical_imaging\MRI\GBM_xCures\meta_data\xcures_annots.xlsx'])
labels.subject_n=categorical(labels.subject_n)
% labels.gender=categorical(labels.gender)
% define three labels
clear new_lb
clc
for m=1:size(labels,1)
    if labels.uncensored(m)==1
    new_lb(m,1)=1;
    elseif labels.uncensored(m)==0 
    new_lb(m,1)=0;
    else
    new_lb(m,1)=nan;
    end

end

labels.Label2=new_lb;

%% 
close all
clc
% 
clear survival coeff coeff2 censored case_id gof_m gof_m2
clear survival coeff coeff2 censored case_id gof_m gof_m2 vols_lim vols_lim_n
close
t = tiledlayout(1,2,'TileSpacing','Compact','Padding','Compact');
for k=3:length(patients)
 clear vols vols3 follow_up data initial_vol vols_n A A2 A_raw A2_raw
 path_mask = dir([path_root_mask,'\',patients(k).name,'\*\MNISpace\All-label_MNI.nii.gz']);
 path_mask.folder;
 row=annots.patient_ID==patients(k).name & annots.Pre_operativeScans=='YES';
 rows=row==true;
 data=annots(rows,:)

 case_id(k-2,1)=categorical(cellstr(patients(k).name));
 

 if size(data,1)<3
     continue
 end
 age_sur(k-2,1)=labels.age_diagnosis(k-2);
 row=labels.subject_n==patients(k).name ;
 rows=row==true;
 labeled=labels(rows,:);

 % extract volume/morphology features
 for m=1:size(data,1)
 
         follow_up(m)=data.weeks_diganosis_to_current_scan(m); 
         if data.Pre_operativeScans(m)=='0'
             vols(m,1)=nan;
             vols3(m,1)=nan;
             vols_n(m,1)=nan;
             continue 
         else  
             mask=niftiread([path_root_mask,'\',patients(k).name,'\',data.timepoint{m},'\MNISpace\All-label_MNI.nii.gz']);
             %volumetric changes
             id1=find(mask==3);
             id3=find(mask==1);
             %vols(m,1)=numel(id1);
             vols3(m,1)=numel(id3);
             
             if numel(id1)>0
             %Morphology
             unique(mask);
             regions=regionprops3(mask,'all')
%              vols(m,1)=regions.ConvexVolume(3);
%              vols(m,1)=regions.Solidity(3);
%              vols(m,1)=regions.SurfaceArea(3);
%              vols(m,1)=regions.EquivDiameter(3);   
               vols(m,1)=regions.Extent(3);
             else
              vols(m,1)=0;
             
             end
             
             vols_n(m,1)=(vols(m)./(vols3(m)+vols(m)))*100;   
         end    

         if m==1
             initial_vol=vols_n(m,1);
         end

 end
 
ws=8;
tResampled=0:ws:follow_up(end);

if size(data,1)==3

vols_interp= interp1(follow_up,vols,tResampled,'linear');
ids=find(vols_interp<0);
%vols_interp(ids)=0;
vols_interp_n= interp1(follow_up,vols_n,tResampled,'linear');
% ids=find(vols_interp<0);
% vols_interp_n(ids)=0;

elseif size(data,1)>=3
vols_interp= interp1(follow_up,vols,tResampled,'makima');
ids=find(vols_interp<0);
%vols_interp(ids)=0;
vols_interp_n= interp1(follow_up,vols_n,tResampled,'linear');
% ids=find(vols_interp<0);
% vols_interp_n(ids)=0;
end 

A_raw=[[follow_up';tResampled'],[vols;vols_interp']];
A_raw=sortrows(A_raw);
A2_raw=[[follow_up';tResampled'],[vols_n;vols_interp_n']];
A2_raw=sortrows(A2_raw);

A=[] ;A2=[];
A(:,1)=A_raw(~isnan(A_raw(:,2)),1);
A(:,2)=A_raw(~isnan(A_raw(:,2)),2);
% A(:,2)=A(:,2)+min(A(:,2));

A2(:,1)=A2_raw(~isnan(A2_raw(:,2)),1);
A2(:,2)=A2_raw(~isnan(A2_raw(:,2)),2);
% A2(:,2)=A2(:,2)+min(A2(:,2));

clear fitobject gof fitobject2 gof2
[fitobject,gof] = fit(A(:,1),A(:,2),"poly2") ;
[fitobject2,gof2] = fit(A2(:,1),A2(:,2),"poly2") ;

nexttile(1)
if  labeled.Label2(1)==0
     plot(follow_up,vols,'.r','MarkerSize',12); hold on
     title('Ehancing volume'); set(gca,'FontSize',12); 
     xlabel('weeks'); ylabel('Volume mm3'); grid on; hold on;
     plot(fitobject,'r')
 
elseif labeled.Label2(1)==1

      plot(follow_up,vols,'.b','MarkerSize',12); hold on
      set(gca,'FontSize',12); 
      xlabel('weeks'); ylabel('Volume mm3'); grid on; hold on;
      plot(tResampled,vols_interp,'.b','MarkerSize',10)
      plot(fitobject,"b--")
      ylim([min(vols) max(vols)])
end
vols_lim(k,:)=[min(vols),max(vols)];
ylim([min(vols_lim(:,1)),max(vols_lim(:,2))])


nexttile(2)
if  labeled.Label2(1)==0
    plot(follow_up,vols_n,'.r','MarkerSize',12); hold on
    title('Percentage'); set(gca,'FontSize',12); 
    xlabel('weeks'); ylabel('Volume mm3'); grid on; hold on;
    plot(fitobject2,"r");hold on
    

elseif labeled.Label2(1)==1
    plot(follow_up,vols_n,'.b','MarkerSize',12); hold on
    set(gca,'FontSize',12); grid on; hold on;
    plot(tResampled,vols_interp_n,'.b','MarkerSize',10)
    plot(fitobject2,"b");hold on

end 

vols_lim_n(k,:)=[min(vols_n),max(vols_n)];
ylim([min(vols_lim_n(:,1)),max(vols_lim_n(:,2))]);

    coeff(k-2,1)= fitobject.p1;
    coeff(k-2,2)= fitobject.p2;
    coeff(k-2,3)= fitobject.p3;
%     coeff2(k,4)= fitobject.p4;

    coeff2(k-2,1)= fitobject2.p1;
    coeff2(k-2,2)= fitobject2.p2;
    coeff2(k-2,3)= fitobject2.p3;
%     coeff2(k,4)= fitobject.p4;
    gof_m(k-2,1)=gof.rsquare;
    gof_m2(k-2,1)=gof2.rsquare;

    
    survival(k-2,1)=labeled.follow_up_from_diagnosis_weeks(1);
    censored(k-2,1)=labeled.uncensored(1);

  
    if strcmp(labeled.gender{1},'male')
            gender(k-2,1)=0;
           
    elseif strcmp(labeled.gender{1},'female')
            gender(k-2,1)=1;
    end 
    
%    

disp('Patient')
k
patients(k).name

pause(1e-10)
end 

%%
clc
T=table(case_id,censored,survival,coeff(:,1),coeff(:,2),coeff(:,3))
T2=table(case_id,censored,survival,coeff2(:,1),coeff2(:,2),coeff2(:,3))
GoF=table(gof_m,gof_m2)
writetable(T,'Poly2_Extent_xcures.xlsx','Sheet','FeatureMatrix')
writetable(T2,'Poly2_Extent_xcures.xlsx','Sheet','FeatureMatrix2')
writetable(GoF,'Poly2_Extent_xcures.xlsx','Sheet','gof')


