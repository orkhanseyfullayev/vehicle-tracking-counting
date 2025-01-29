# Vehicle Tracking and Counting System

A real-time vehicle tracking and counting system using **YOLOv8** and **OpenCV**. This project processes video footage to detect and count vehicles crossing predefined regions.

---

## Features
- **Real-time vehicle detection** using YOLOv8.
- **Vehicle counting** based on predefined regions.
- **Supports multiple vehicle classes** (e.g., cars, trucks).
- **Customizable detection regions** to track movement in specific areas.

---

## Requirements

Ensure you have the following installed:
- Python 3.7+
- OpenCV
- UltraLytics YOLO

Install required Python packages:
```bash
pip install ultralytics opencv-python numpy
```

---

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/vehicle-tracking-counting.git
   cd vehicle-tracking-counting
   ```

2. **Prepare your video file:**
   - Place your video in the `Videos/` directory.
   - Ensure the filename matches what you enter when running the script.

3. **Run the program:**
   ```bash
   python main.py
   ```

4. **Vehicle Counting Display:**
   - Vehicles will be detected and counted in predefined regions.
   - The count will be displayed on the video feed.

---

## File Structure

```
📂 vehicle-tracking-counting
├── main.py           # Main script for vehicle tracking and counting
├── README.md         # Project documentation
```

---

## Customization

- **Detection Threshold:**
  Modify the `score` threshold in `main.py` to adjust the sensitivity of vehicle detection.
  ```python
  if score < 0.5:
      continue
  ```

- **Counting Regions:**
  Adjust the coordinates of `region_left` and `region_right` to define new counting areas.
  ```python
  region_left = np.array([(0, 360), (640, 360), (640, 370), (0, 367)])
  ```

---

## Future Improvements
- Adding tracking IDs for each vehicle.
- Exporting count data to a CSV or database.
- Enhancing detection accuracy with model fine-tuning.

---

## License
This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and share it.

---

## Author
Developed by **[Your Name](https://github.com/yourusername)**. For queries, reach out via email or GitHub.
