#!/bin/bash


# All files
docker run --rm --gpus '"device=3"' -v /raid/andek67/IMPACT_demonstrator/testsubject:/in -v /raid/andek67/IMPACT_demonstrator/testoutput:/out kerasdicom

mv /raid/andek67/IMPACT_demonstrator/testoutput/segmentation.nii.gz /raid/andek67/IMPACT_demonstrator/testoutput/segmentation_7channels.nii.gz
mv /raid/andek67/IMPACT_demonstrator/testoutput/segmentation_postp.nii.gz /raid/andek67/IMPACT_demonstrator/testoutput/segmentation_postp_7channels.nii.gz

# Only T1 GD
docker run --rm --gpus '"device=3"' -v /raid/andek67/IMPACT_demonstrator/testsubject_T1GD:/in -v /raid/andek67/IMPACT_demonstrator/testoutput:/out kerasdicom

mv /raid/andek67/IMPACT_demonstrator/testoutput/segmentation.nii.gz /raid/andek67/IMPACT_demonstrator/testoutput/segmentation_1channels.nii.gz
mv /raid/andek67/IMPACT_demonstrator/testoutput/segmentation_postp.nii.gz /raid/andek67/IMPACT_demonstrator/testoutput/segmentation_postp_1channels.nii.gz

# T1 GD and qMRI GD
docker run --rm --gpus '"device=3"' -v /raid/andek67/IMPACT_demonstrator/testsubject_T1GD_qMRIGD:/in -v /raid/andek67/IMPACT_demonstrator/testoutput:/out kerasdicom

mv /raid/andek67/IMPACT_demonstrator/testoutput/segmentation.nii.gz /raid/andek67/IMPACT_demonstrator/testoutput/segmentation_4channels.nii.gz
mv /raid/andek67/IMPACT_demonstrator/testoutput/segmentation_postp.nii.gz /raid/andek67/IMPACT_demonstrator/testoutput/segmentation_postp_4channels.nii.gz
