#!/bin/bash

# Convert DICOM(s) to nifti(s)
for folder in /nifti_in/*; do
  python /home/convert_to_nifti.py $folder /nifti_in/
done

#cp /in/* /nifti_in

# Perform segmentation
echo "Performing segmentation"
python /home/predict.py  /nifti_in/ /nifti_out/segmentation.nii.gz

# Post-process segmentations
echo "Post-processing segmentation"
python /home/post_process_segmentation.py /nifti_out/segmentation.nii.gz /nifti_out/segmentation.nii.gz

# Convert nifti segmentation to DICOM RTSTRUCT
echo "Converting segmentation to RTSTRUCT"
python /home/convert_to_RTSTRUCT.py /nifti_out/segmentation.nii.gz /in /out

# copy nifti output to output directory
#cp /nifti_out/*.nii.gz /out 

# copy original nifti to output directory
#cp /nifti_in/*.nii.gz /out 
