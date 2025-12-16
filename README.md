# Vehicle Tracking, Speed Estimation, and Google Sheets Logging using OpenCV

## Project Overview
This project is developed as an **OpenCV-based computer vision assignment**.  
It focuses on **vehicle tracking and speed estimation from a road traffic video**, with an additional feature to **log overspeeding vehicles to Google Sheets** for further analysis and record keeping.

The system detects vehicles, estimates their speed using perspective transformation, and automatically uploads data to Google Sheets when a vehicle exceeds a predefined speed limit.

---

## Author Information
- **Name:** Chanikan Noiubon (张敏)  
- **Student ID:** 192364111  

This project is submitted as part of an **OpenCV course/project requirement**.

---

## Key Features
- Vehicle detection and tracking from video input
- Speed estimation using perspective transformation
- Real-world coordinate mapping
- Overspeed detection based on a predefined speed limit
- **Automatic data logging to Google Sheets for overspeeding vehicles**
- Designed for traffic monitoring and intelligent transportation applications

---

## Google Sheets Integration (Overspeed Logging)
When a vehicle is detected with a speed **exceeding the defined speed limit**, the system automatically sends the following information to **Google Sheets**:

- Timestamp
- Vehicle ID
- Estimated speed (km/h)
- message Ex: Overspeed > 70 km/h

This enables:
- Traffic violation recording
- Data analysis and visualization
- Long-term monitoring of speeding behavior

> For security reasons, Google service account credentials are **not included** in this repository.

---

## Requirements
- Python 3.x  
- OpenCV  
- NumPy  
- Google API Client Libraries  
- Internet connection (for Google Sheets logging)

---

## Credentials Setup (Required for Google Sheets Logging)

This project requires a **Google Cloud Service Account** to write data to Google Sheets.

### Steps:
1. Create a Google Cloud service account and enable the **Google Sheets API**.
2. Download the service account key file.
3. Rename it to:

## Run by
```bash
 E:\conda_envs\opencv_car\python.exe script.py --source_video_path "NUAA road 1080p.mp4"
