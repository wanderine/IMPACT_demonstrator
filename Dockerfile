# Start from CentOS 7
FROM nvidia/cuda:10.1-base

# Set the shell to bash
SHELL ["/bin/bash", "-c"]

# Install needed packages
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y wget
RUN apt-get install -y git
RUN apt-get install -y unzip

# Install Anaconda
RUN cd /home && \
  wget https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-x86_64.sh && \
  chmod +x Anaconda3-2019.07-Linux-x86_64.sh && \
  ./Anaconda3-2019.07-Linux-x86_64.sh -b -p ~/anaconda && \
  rm Anaconda3-2019.07-Linux-x86_64.sh

# Add Anaconda to path
ENV PATH=/root/anaconda/bin:$PATH

# Create keras conda environment
COPY kerasdicom.yml /home
RUN conda env create -f /home/kerasdicom.yml

# Add conda environment to path
ENV PATH=/root/anaconda/envs/kerasdicom/bin:$PATH

# Create folders for input and output data
RUN mkdir /{in,out,nifti_in,nifti_out}

# Copy scripts for running segmentation
COPY pipeline.sh /home
COPY predict.py /home
COPY post_process_segmentation.py /home
COPY convert_to_nifti.py /home
COPY convert_to_RTSTRUCT.py /home
COPY utilities.py /home

COPY dcm2niix_afni /home

COPY myWeights_weight60000_depth4_nfilter16_CV3_BRATS_augmented_defaced.h5 /home
COPY myWeights_weight60000_depth4_nfilter16_CV3_BRATS_qMRIGD_augmented_defaced.h5 /home
COPY myWeights_weight60000_depth4_nfilter16_CV3_BRATS_qMRI_qMRIGD_augmented_defaced.h5 /home

RUN chmod +x /home/pipeline.sh

ENTRYPOINT ["/home/pipeline.sh"]
