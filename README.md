# mlmi-project
Self-supervised Multimodal Representation Learning - MLMI SS23

## Papers
https://drive.google.com/drive/folders/1udxSmKlTU1gG0vKMj-kEd91CSBJMI-OF

## ShareLatex
https://sharelatex.tum.de/7138674284xmjqszfcktdq

## Baseline Implementations
### Contrastive Multimodal Representation Learning
- [ImageBind](https://github.com/facebookresearch/ImageBind)
- [CLIP](https://github.com/openai/CLIP)
- [DeCLIP](https://github.com/Sense-GVT/DeCLIP)
- [MS-CLIP](https://github.com/Hxyou/MSCLIP)
- [MaskCLIP](https://github.com/LightDXY/MaskCLIP)
- [CoDi](https://github.com/microsoft/i-Code/tree/main/i-Code-V3)
- [Swin UNETR](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR): Pre-trained CT & MRI Encoders

### Supervised Brain Imaging SOTA (to compare learned representation quality)
- [RadImageNet](https://github.com/BMEII-AI/RadImageNet)
- [MedMNIST](https://medmnist.com/)
- [Brain Tumor Segmentation 2020](https://arxiv.org/abs/2004.10664)
- [Segmentation 2023](https://arxiv.org/abs/2306.03730)
- [The Brain Tumor Segmentation (BraTS) Challenge 2023](https://arxiv.org/abs/2305.09011v3)
- TODO: Neurodegenerative Disease Diagnosis
- TODO: fMRI to Image aka mind reading

### Brain multimodal datasets
- [iEEG-fMRI](https://www.nature.com/articles/s41597-022-01173-0)
- [fMRI-Image](http://naturalscenesdataset.org/)
- [fMRI-Audio](https://www.nature.com/articles/s41597-021-01033-3)
- [MRI-CT-PET](http://www.oasis-brains.org/)
- [MRI-PET-DTI](https://tadpole.grand-challenge.org/Data/)
- TODO: MEG & SPECT
- TODO: https://github.com/sfikas/medical-imaging-datasets

### Reviews
- [Multimodal medical Review 2022](https://drive.google.com/file/d/1Bm9KTSyNnRDZkC6DUCkqacq80k18Jk4X/view?usp=drive_link)


# Plan
- using EEG to learn improved fMRI representations
- use Dings eeg encoder + Svin-UNETR fMRI encoder
- contrastive learning between representations
- test task: alzheimer classification
