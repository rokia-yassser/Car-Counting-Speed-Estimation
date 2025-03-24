# Car Counting and Speed Estimation (YOLOv8, OpenCV)

## 🚀 Project Overview
This project is a **real-time vehicle detection, tracking, and speed estimation** system using **YOLOv8** and **OpenCV**. It detects cars in a video feed, assigns unique IDs to track them, and estimates their speed using centroid tracking and time difference calculations.

## 📌 Features
- **Real-time car detection** using YOLOv8.
- **Centroid tracking** to assign unique IDs to vehicles.
- **Car counting** when vehicles cross a predefined line.
- **Speed estimation** in km/h using Euclidean distance and time tracking.
- **Overlay display** showing detected cars, speed, and total count.

## 🛠️ Installation & Setup
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/Car-Counting-Speed-Estimation.git
cd Car-Counting-Speed-Estimation
```

### 2️⃣ Install Dependencies
```bash
pip install ultralytics opencv-python numpy
```

### 3️⃣ Run the Project
```bash
python car_counting.py
```

## 📂 Project Structure
```
📁 Car-Counting-Speed-Estimation
│── 📜 car_counting.py  # Main Python script
│── 📜 README.md         # Project documentation
```

## ⚡ How It Works
1. **YOLOv8 detects cars** in the video.
2. **Centroid tracking assigns unique IDs** to each car.
3. **Each car’s speed is estimated** based on its movement between frames.
4. **A counting line detects cars crossing** and increments the count.

## 📊 Example Output
- **Bounding boxes** around detected cars.
- **Car speed (km/h) displayed** on top of each vehicle.
- **Total cars counted shown** on the screen.

## 🎯 Next Steps
- [ ] Improve speed estimation with camera calibration.
- [ ] Save detected vehicle speeds to a CSV file.
- [ ] Deploy as a web application using FastAPI or Flask.

## 🤝 Contributing
Feel free to fork this repository, open issues, and submit pull requests!


