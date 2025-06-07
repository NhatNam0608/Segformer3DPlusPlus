
# Medical Image Segmentation

This repository contains the implementation of the **Medical Image Segmentation** project. In this project, we research and develop **SegFormer3D++**, a deep learning model for 3D medical image segmentation.

---

## 🚀 Getting Started

Follow the steps below to set up and run the project:

### 1. Create and activate a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate     
```

### 2. Install dependencies:
```bash
pip install -r requirements.txt
```
### 2. Download data:
Download data for this repository at ...
Structure like this image:
![image](https://github.com/user-attachments/assets/60c55384-8e4a-454e-b7b3-32f0b21b98e9)

---

## 🧪 How to Evaluate SegFormer3D++

### On the **BraTS** dataset:

- **BSM** evaluation:
```bash
python main.py --dataset brats --type bsm --command eval
```

- **EPA** evaluation:
```bash
python main.py --dataset brats --type epa --command eval
```

### On the **ACDC** dataset:

- **BSM** evaluation:
```bash
python main.py --dataset acdc --type bsm --command eval
```

- **EPA** evaluation:
```bash
python main.py --dataset acdc --type epa --command eval
```

---

## 🔍 How to Run Inference with SegFormer3D++

### On the **BraTS** dataset:

- **BSM** inference:
```bash
python main.py --dataset brats --type bsm --command inference
```

- **EPA** inference:
```bash
python main.py --dataset brats --type epa --command inference
```

### On the **ACDC** dataset:

- **BSM** inference:
```bash
python main.py --dataset acdc --type bsm --command inference
```

- **EPA** inference:
```bash
python main.py --dataset acdc --type epa --command inference
```
## 🔍 How to Run Application with SegFormer3D++
```bash
python app.py
```
