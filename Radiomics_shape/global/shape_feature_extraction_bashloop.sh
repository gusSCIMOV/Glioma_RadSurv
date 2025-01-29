#!/bin/bash
#This script calls shape_feature_extraction.sh for all patients in the specified directory
# set mother_path (functions) ANTS path (AP) String of MRI sequence (IMSTR) and mask (MASK)
database=xCures

IMSTR=T1_C_SRI24_SkullS_BiasCorrect_IntStnd
MASK=01_tumor_segmentation_brats_SRI24_label_4
tp=Baseline
#habitat=whole_tumor
#habitat=subregions

#####################################################################

root_path=/app/Data/_Brain/Radiology/_Adult/_Glioma/${database}/Preprocessed/Pre-op-Post-op
directory=${root_path}
out=/app/Data/_Brain/Radiology/_Adult/_Glioma/${database}/Feature_extraction/Radiomics_features
mother_path=/app/Codebase/gustavo_script/_Neuroimaging/Radiomics_Features_extraction/Radiomics_shape/global
AP=/app/IDIA/MRI/Libraries_tools/Libraries/ANTS/build/bin

###########################################################

SCRIPT=${mother_path}/shape_feature_extraction.sh
PROG=${mother_path}/ShapeFeatures3D_HPC/build/ShapeFeatures3D

SC=$(echo "$SCRIPT" | sed 's/.*\/\(.*\)\..*/\1/')
OUTPUT_LOG=$SC"_$IMSTR"

mkdir ${mother_path}/LOG
LOGDIR=${mother_path}/LOG/${OUTPUT_LOG}

if [[ -d $LOGDIR ]]; then
    rm -Rf $LOGDIR/*
    rmdir $LOGDIR
fi


echo $LOGDIR
mkdir $LOGDIR

DATA=${directory}
cd $DATA
echo "current directory .... $PWD"

folderarray=(`ls`)

echo "ls ... $folderrray "

for ((j=0; j<${#folderarray[@]}; j++))

do
    echo "herein........$j"
    cd $LOGDIR

    input_im=$DATA/${folderarray[j]}/${tp}/${folderarray[j]}_${IMSTR}.nii.gz
    input_m=$DATA/${folderarray[j]}/${tp}/${folderarray[j]}_${MASK}.nii.gz
    out_path=${out}/${folderarray[j]}/${tp}/shape_global

    echo "label $MASK .......... ${input_m} "
    echo "MRI sequence ........ ${input_im}"

    if [ -f $input_m ]; then
        export j=$j DATA=$DATA PROG=$PROG AP=$AP patient=${folderarray[j]} tp=$tp MASK=$MASK
        echo "MRI sequence ........ ${input_im}"
        echo "Mask ......... ${input_m} "
        echo "out_path ......... ${out_path} "
        bash $SCRIPT $input_im $input_m $out_path 
        echo "..........................................................................................................."
    fi
done

cd ${mother_path}
#output_name=shape_${IMSTR}_${directory}
#tar -czf ${output_name}.tar.gz ${directory}/
