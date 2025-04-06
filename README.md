# Covid-19-Xray-classifier
This is a ML application which uses PyTorch to classify Covid-19 chest X-ray images for COVID-19 detection. Its purpose is to differentiate between normal, viral pneumonia, and COVID-19 radiographs. Built for medical image analysis research and educational purposes.

## Prerequisites

Before starting, ensure you have these installed:

    Python 3.7+

    pip/conda

    Git (optional but recommended)




## 1. Install Python and Libraries:

```
pip install torch torchvision numpy pandas matplotlib pillow scikit-learn
```




## 2. Folder Structure for D:\covid_xray_classifier

```
D:\covid_xray_classifier\
│   covid_classifier.py    # The main script
│
└───data\                  # Main data directory
    ├───normal\            # Directory for normal X-ray images
    │       normal1.jpg    # Example normal X-ray image
    │       normal2.jpg
    │       ...
    │
    ├───viral\             # Directory for viral pneumonia X-ray images
    │       viral1.jpg     # Example viral pneumonia X-ray image
    │       viral2.jpg
    │       ...
    │
    └───covid\             # Directory for COVID-19 X-ray images
            covid1.jpg     # Example COVID-19 X-ray image
            covid2.jpg
            ...
```