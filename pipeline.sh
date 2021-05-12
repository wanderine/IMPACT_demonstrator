#!/bin/bash

ls /in

cp -r /in/* /nifti_in/

# Convert DICOM(s) to nifti(s)
for folder in /nifti_in/*; do
  echo $folder 
  /home/dcm2niix_afni -o $folder -f volume $folder
done

ls /nifti_in/
ls /nifti_in/T1GD/

# Perform segmentation
echo "Performing segmentation"
python /home/predict.py  /nifti_in /nifti_out/segmentation.nii.gz

# Post-process segmentations
echo "Post-processing segmentation"
python /home/post_process_segmentation.py /nifti_out/segmentation.nii.gz /nifti_out/segmentation_postp.nii.gz

# Convert nifti segmentation to DICOM RTSTRUCT
echo "Converting segmentation to RTSTRUCT"
python /home/convert_to_RTSTRUCT.py /nifti_out/segmentation_postp.nii.gz /in/T1GD /out
python /home/read_RTSS.py -in /out/segmentationRTSTRUCT.dcm --out /out/RTSS_info.json

# copy nifti output to output directory
cp /nifti_out/segmentation* /out 

# copy original nifti to output directory
cp /nifti_in/T1GD/*.nii /out/T1GD.nii

cp /nifti_in/qMRIT1GD/*.nii /out/qMRIT1GD.nii

cp /nifti_in/qMRIT2GD/*.nii /out/qMRIT2GD.nii

cp /nifti_in/qMRIPDGD/*.nii /out/qMRIPDGD.nii



