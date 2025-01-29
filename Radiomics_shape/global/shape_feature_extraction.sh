#!/bin/bash
#This script computes global shape features. Check out ReadMe.txt in /ShapeFeatures3D_HPC for more information.

cd $DATA
CASES=(`ls`)
echo "Subject ID: ${CASES[$j]}"

vol=$1 #input file
echo "Intensity Image: $vol"
mask=$2 #input file
echo "Label Image: $mask"
OUTPUT_PATH=$3 #output path
echo "OUTPUT_PATH: $OUTPUT_PATH"
# habitat=$4
# echo "Tumor habitat: $habitat"

if [[ ! -d $OUTPUT_PATH ]]; then
    mkdir -p $OUTPUT_PATH
fi


echo "Check image extensions"
echo "Checking image extension .nii or .nii.gz"
img_ext=$(echo "$vol" | sed 's/.*\.*\.\(.*\)/\1/')
if [ "$img_ext" == "gz" ]; then
    OUTPUT_PATH1=$(echo "$vol" | sed 's/\(.*\/\).*\..*\..*/\1/')
    outim=$(echo "$vol" | sed 's/.*\/\(.*\)\..*\..*/\1/')
    #echo "OUTPUT PATH: $OUTPUT_PATH1"
else
    OUTPUT_PATH1=$(echo "$vol" | sed 's/\(.*\/\).*\..*/\1/')
    outim=$(echo "$vol" | sed 's/.*\/\(.*\)\..*/\1/')
    #echo "OUTPUT PATH: $OUTPUT_PATH1"
fi

#Merge all existing labels to compute shape descriptors of the entire tumor volume (ignore line below if not necessary)
#$AP/ImageMath 3 $mask ReplaceVoxelValue $mask 1 4 1

#Call to Extract 3D Features	
#<$PATH/ShapeFeatures3D> <LabelImage> <IntensityImage> <OutputFeatureFile.txt> -- this text file will contain feature values for each individual label present in the mask
outname=$OUTPUT_PATH/${patient}_${MASK}\.txt
if [ ! -f $outname ]; then
    $PROG $mask $vol $outname

else
    echo "shape features already exist: $outname"

fi


#Merge all existing labels to compute shape descriptors of the entire tumor volume (ignore line below if not necessary)
#$AP/ImageMath 3 $OUTPUT_PATH/Label_habitat.nii.gz ReplaceVolxelValue $mask 1 4 1

#Call to Extract 3D Features -- this time, the text file will contain feature values for the entire tumor habitat only 
#$PROG $OUTPUT_PATH/Label_habitat.nii.gz $vol $OUTPUT_PATH/shape_feats_habitat\.txt
