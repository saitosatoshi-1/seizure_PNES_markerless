# Markerless Video Analysis for Epileptic Seizures vs PNES  
Quantitative comparison of motor patterns in **epileptic focal-to-bilateral tonic–clonic seizures (FBTCS)** and **psychogenic nonepileptic seizures (PNES)** using **markerless video analysis**.  
This repository provides reproducible Python code for:

- Skeleton-based movement analysis  
- Silhouette segmentation using **Ultralytics YOLO11-seg**
- Optical-flow–based motion quantification  
- QuickTime-compatible MP4 export suitable for clinical and academic use  

This project supports the development of objective, interpretable biomarkers for differentiating epileptic seizures from PNES using only video recordings.

---

## 🚀 Features
- **YOLO11 segmentation** for silhouette extraction  
- **Frame-rate normalization** and preprocessing  
- **QuickTime-safe H.264 re-encoding**  
- Modular Python script with CLI  
- Jupyter/Colab-friendly demo notebook  

---

## 🔧 Installation
Install required packages:
pip install ultralytics opencv-python numpy scipy pandas matplotlib

---

## ▶️ Usage

1. Basic usage (command line)
python src/silhouette_yolo.py --input your_video.mp4 --output silhouette_output.mp4

2. Arguments
The script performs:
	1.	Video loading
	2.	YOLO segmentation
	3.	Visualization overlay
	4.	QuickTime-compatible H.264 encoding

---

## 🧠 Method Summary

This repository is part of a research project comparing seizure motor patterns between FBTCS and PNES using:
	•	Skeleton velocity vectors
	•	Silhouette optical-flow energy
	•	Amplitude decay slopes (ADS)
	•	Periodicity measures (Kendall’s τ)
	•	Shannon entropy

All methods follow the analysis pipeline used in the associated manuscript.

---

## 👤 Author

Satoshi Saito, MD
Department of Epileptology, National Center of Neurology and Psychiatry
Tokyo, Japan
