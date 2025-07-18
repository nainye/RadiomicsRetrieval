# RadiomicsRetrieval: A Customizable Framework for Medical Image Retrieval Using Radiomics Features

:fire:  Official implementation of "RadiomicsRetrieval: A Customizable Framework for Medical Image Retrieval Using Radiomics Features" (MICCAI 2025)

[![arXiv](https://img.shields.io/badge/arXiv-2507.08546-red)](https://arxiv.org/pdf/2507.08546.pdf)

![model](images/model_architecture.jpg)

## Datasets

This project utilizes publicly available medical imaging datasets. You can download them from the following sources:

### :lungs: Lung Tumor CT
- [NSCLC-Radiomics](https://www.cancerimagingarchive.net/collection/nsclc-radiomics/)
- [NSCLC-Radiomics-Interobserver1](https://www.cancerimagingarchive.net/collection/nsclc-radiomics-interobserver1/)
- [RIDER-LungCT-Seg](https://www.cancerimagingarchive.net/collection/rider-lung-ct/)
- [NSCLC Radiogenomics](https://www.cancerimagingarchive.net/collection/nsclc-radiogenomics/)
- [LUNG-PET-CT-Dx](https://www.cancerimagingarchive.net/collection/lung-pet-ct-dx/)
  
### :brain: Brain Tumor MRI
- [Adult Glioma (GLI)](https://www.synapse.org/Synapse:syn51156910/wiki/622351)
- [Meningioma (MEN)](https://www.synapse.org/Synapse:syn51156910/wiki/622353)


## Usage

### Installation

To set up your development environment, follow the steps below:

1. **Pull the Docker image:**

    We are using the `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel` Docker image. You can pull it from Docker Hub by running:

    ```sh
    docker pull pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel
    ```

2. **Run the Docker container:**

    Start a container from the pulled image. You can mount your project directory into the container for easy development:

    ```sh
    docker run --shm-size=192gb --gpus=all -it -v /path/to/your/project/:/workspace --name radiomicsretrieval pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel /bin/bash
    ```

    Replace `/path/to/your/project` with the actual path to your project directory.

3. **Set up the Python environment:**

    Inside the container, clone this repository and install the required dependencies:

    ```sh
    git clone https://github.com/nainye/RadiomicsRetrieval.git && cd RadiomicsRetrieval
    pip install -r requirements.txt
    ```

### Data Preprocessing
To preprocess the datasets, run the following notebooks:

#### :lungs: Lung Tumor CT (NSCLC)
  [source/preprocessing_NSCLC.ipynb](source/preprocessing_NSCLC.ipynb)

#### :brain: Brain Tumor MRI (BraTS)    
  Preprocessing notebook is under construction and will be uploaded soon.

### Model Training

#### Common Setup

Before training the model, please download the SAM-Med3D checkpoint and place it as follows:

  - 🔗 [Download `sam_med3d_turbo.pth`](https://huggingface.co/blueyo0/SAM-Med3D/blob/main/sam_med3d_turbo.pth)
  - Save it to:
     ```
     ./source/sam_med3d_turbo.pth
     ```
#### :lungs: Lung Tumor CT (NSCLC)

To train the model on NSCLC CT data, run the following script:

  ```sh
  ./scripts/run_train_RadiomicsRetireval_NSCLC_Img+Rad_withAPE.sh
  ```

     
#### :brain: Brain Tumor MRI (BraTS)  
Training code for brain tumor retrieval is under construction and will be uploaded soon.

  
## Citation
If you use this code for your research, please cite our papers.

**BibTeX:**
```bibtex
@article{na2025radiomicsretrieval,
  title={RadiomicsRetrieval: A Customizable Framework for Medical Image Retrieval Using Radiomics Features},
  author={Na, Inye and Rue, Nejung and Chung, Jiwon and Park, Hyunjin},
  journal={arXiv preprint arXiv:2507.08546},
  year={2025}
}
```

## Contact

For any inquiries or support, please contact us at `niy0404@gmail.com`.
