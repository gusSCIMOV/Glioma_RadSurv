% % intensiity standardization with linear scaling LUMIERE

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
mother_path='H:\MRI\Libraries_tools\Preprocessing\rename_files_clean_mask';
addpath(genpath(mother_path))
im_mod='CT1'
t_p='01'
% Generate a template from N randomly chosen subjects
%
% XCURES
% path_root=['C:\Users\GXP013\Documents\Medical_imaging\MRI\GBM_xCures\Preprocessing\Total_pp_results\Nifti\'];
% path_root_mask=['C:\Users\GXP013\Documents\Medical_imaging\MRI\GBM_xCures\Preprocessing\Total_pp_results\Nifti\'];

% LUMIERE (seg 1 >>enhacing tumor,  2 >>necrosis, 3>>> edema)

path_root=['\\onfnas01.uwhis.hosp.wisc.edu\radiology\Groups\IDIAGroup\Stephen_Transfers\LUMIERE\'];
path_root_mask=['\\onfnas01.uwhis.hosp.wisc.edu\radiology\Groups\IDIAGroup\',........
    'Data\_Brain\Radiology\_Adult\_Glioma\LUMIERE\Raw_imaging\imaging'];


addpath(genpath(path_root_mask))
patients=dir(path_root_mask)

clc
addpath(['H:\MRI\GBM\LUMIERE\metadata\'])
annots=readtable(['H:\MRI\GBM\LUMIERE\metadata\QC_Lumiere.xlsx'])
annots.Patient=categorical(annots.Patient)
annots.DeepBraTumIA=categorical(annots.DeepBraTumIA)
labels=readtable(['H:\MRI\GBM\LUMIERE\metadata\LUMIERE-labels.csv'])
labels.Patient=categorical(labels.ID1)
labels.SurvivalTime_weeks_=labels.OS_weeks;
% define three labels
clear new_lb
clc
for m=1:size(labels,1)
    if labels.SurvivalTime_weeks_(m) <72.5
    new_lb(m,1)=0;
    elseif labels.SurvivalTime_weeks_(m) >= 72.5 
    new_lb(m,1)=1;
    end

end

labels.Label2=new_lb;
labels.AgeAtSurgery_years_=labels.Age;
%% 
close all
clc
% 
clear survival coeff coeff2 censored case_id gof_m gof_m2 vols_lim vols_lim_n
close
t = tiledlayout(1,2,'TileSpacing','Compact','Padding','Compact');

all_measures=[]

for k=3:length(patients)
    
 if ~ismember(patients(k).name,labels.Patient)
     continue
 end 
    
    
 clear vols0 vols vols3 follow_up data initial_vol vols_n A A2 A_raw A2_raw mask regions
 clear vols1 vols2 vols3 vols4 vols5 vols6 vols7 vols8
 path_mask = dir([path_root_mask,'\',patients(k).name,'\*\DeepBraTumIA-segmentation\atlas\segmentation\seg_mask*']);
 path_mask.folder;
 row=annots.Patient==patients(k).name & annots.DeepBraTumIA~='0';
 rows=row==true;
 data=annots(rows,:)

 case_id(k-2,1)=categorical(cellstr(patients(k).name));
 
 
 if size(data,1)<3
     continue
 end
 age_sur(k-2,1)=labels.AgeAtSurgery_years_(k-2);
 row=labels.Patient==patients(k).name;
 rows=row==true;
 labeled=labels(rows,:);

 % extract volume/morphology features
 for m=1:length(path_mask)
 
         follow_up(m)=data.weeks(m); 
         
         if categorical(data.DeepBraTumIA(m))=='0'
             vols(m,1)=nan;
             vols3(m,1)=nan;
             vols_n(m,1)=nan;
             continue 
         else  
             mask=niftiread([path_mask(m).folder,'\',path_mask(m).name]);
             %volumetric changes
             id1=find(mask==1);
             id2=find(mask==2);
             id3=find(mask==3);
             vols0(m,1)=numel(find(mask>0));
             vols1(m,1)=numel(id1);
             vols2(m,1)=numel(id2);
             vols3(m,1)=numel(id3);
             
             if numel(id3)>0
                 %Morphology
                 unique(mask);
                 regions=regionprops3(mask,'all');
                 vols4(m,1)=regions.ConvexVolume(1);
                 vols5(m,1)=regions.Solidity(1);
                 vols6(m,1)=regions.SurfaceArea(1);
                 vols7(m,1)=regions.EquivDiameter(1);   
                 vols8(m,1)=regions.Extent(1)
             elseif numel(id1)==0
                    continue
             end
             
             vols_n(m,1)=(vols1(m)./(vols3(m)+vols1(m)))*100; 
         end    

         if m==1
             initial_vol=vols_n(m,1);
         end

 end
data.morphology0=vols0;
data.morphology1=vols1;
data.morphology2=vols2;
data.morphology3=vols3;
data.morphology4=vols4;
data.morphology5=vols5;
data.morphology6=vols6;
data.morphology7=vols7;
data.morphology8=vols8;

% if k>3
    all_measures=[all_measures;data]
% else
% end


ws=8
tResampled=0:ws:follow_up(end);

if size(data,1)==3

vols_interp= interp1(follow_up,vols1,tResampled,'linear');
ids=find(vols_interp<0);
%vols_interp(ids)=0;
vols_interp_n= interp1(follow_up,vols_n,tResampled,'linear');
% ids=find(vols_interp<0);
% vols_interp_n(ids)=0;

elseif size(data,1)>=3
vols_interp= interp1(follow_up,vols1,tResampled,'makima');
ids=find(vols_interp<0);
%vols_interp(ids)=0;
vols_interp_n= interp1(follow_up,vols_n,tResampled,'linear');
% ids=find(vols_interp<0);
% vols_interp_n(ids)=0;
end 

A_raw=[[follow_up';tResampled'],[vols1;vols_interp']];
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
     plot(follow_up,vols1,'.r','MarkerSize',12); hold on
     title('Ehancing volume'); set(gca,'FontSize',12); 
     xlabel('weeks'); ylabel('Volume mm3'); grid on; hold on;
     plot(fitobject,'r')

elseif labeled.Label2(1)==1

      plot(follow_up,vols1,'.b','MarkerSize',12); hold on
      set(gca,'FontSize',12); 
      xlabel('weeks'); ylabel('Volume mm3'); grid on; hold on;
      plot(tResampled,vols_interp,'.b','MarkerSize',10)
      plot(fitobject,"b--")
      ylim([min(vols1) max(vols1)])
end
vols_lim(k,:)=[min(vols1),max(vols1)];
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
    
    if ~isnan(labels.SurvivalTime_weeks_(k-2))
            disp('no censored')
            survival(k-2,1)=labels.SurvivalTime_weeks_(k-2);
            censored(k-2,1)=0;
    elseif isnan(labels.SurvivalTime_weeks_(k-2))
            survival(k-2,1)=follow_up(end);
            censored(k-2,1)=1;
    end 

    if strcmp(labels.Sex{k-2},'male')
            gender(k-2,1)=0;
           
    elseif strcmp(labels.Sex{k-2},'female')
            gender(k-2,1)=1;
    end 

%    
disp('survival')
labels.SurvivalTime_weeks_(k-2)
disp('label')
labeled.Label2(1)
disp('Patient')
k
patients(k).name

pause(10e-10)
end 

GoF=table(age_sur,gender)
writetable(GoF,'demographics_lum.xlsx','Sheet','FeatureMatrix')

T=table(case_id,censored,survival,coeff(:,1),coeff(:,2),coeff(:,3))
T2=table(case_id,censored,survival,coeff2(:,1),coeff2(:,2),coeff2(:,3))
GoF=table(gof_m,gof_m2)
writetable(T,'Poly2_vols_lum.xlsx','Sheet','FeatureMatrix')
writetable(T2,'Poly2_vols_lum.xlsx','Sheet','FeatureMatrix2')
writetable(GoF,'Poly2_vols_lum.xlsx','Sheet','gof')



