#!/bin/bash

ls /in

cp -r /in/* /nifti_in/

# Convert DICOM(s) to nifti(s)
for folder in /nifti_in/*; do
  echo $folder 
  #mkdir /nifti_in/$folder   
  python /home/convert_to_nifti.py $folder $folder
done

ls /nifti_in/
ls /nifti_in/T1GD/

#cp /in/* /nifti_in

# Perform segmentation
echo "Performing segmentation"
python /home/predict.py  /nifti_in /nifti_out/segmentation.nii.gz

# Post-process segmentations
echo "Post-processing segmentation"
python /home/post_process_segmentation.py /nifti_out/segmentation.nii.gz /nifti_out/segmentation.nii.gz

# Convert nifti segmentation to DICOM RTSTRUCT
echo "Converting segmentation to RTSTRUCT"
python /home/convert_to_RTSTRUCT.py /nifti_out/segmentation.nii.gz /in/T1GD /out

# copy nifti output to output directory
cp /nifti_out/segmentation.nii.gz /out 

# copy original nifti to output directory
cp /nifti_in/T1GD/*.nii.gz /out 
