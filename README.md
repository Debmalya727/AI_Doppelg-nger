# 🧠 AI Doppelgänger Detection System

An advanced AI-powered system that detects **lookalike faces (doppelgängers)** and identifies **potential impersonator profiles** using face recognition, image enhancement, and social media profile analysis.

---

## 🚀 Features

### 🔹 Core Face Matching
- Uses **InsightFace** for high-accuracy embeddings
- Cosine similarity-based matching
- Robust against:
  - Tilted images
  - Blurry/compressed images
  - Lighting variations

### 🔹 Image Enhancement
- Histogram equalization
- Sharpening filters
- Face alignment using landmarks
- Augmented comparison (rotation + scaling)

### 🔹 Multi-Face Detection
- Detect multiple faces in one image
- Assigns face numbers
- Matches each face independently

### 🔹 Web Application (Flask)
- Upload image via browser
- Webcam capture support (start/stop)
- Annotated face display
- Match results with confidence scores

### 🔹 Smart Result Handling
- Best-match filtering (removes duplicates)
- Confidence visualization
- CSV logging of results

### 🔹 Instagram Profile Integration (Phase 3)
- Fetch profile data using **Instaloader**
- Supports:
  - Username
  - Profile picture
  - Bio
- Hybrid login using cookies + browser
- Profile-based matching (face + name + bio)

---

## 🎯 Project Goal

To build a system that can:
- Detect **doppelgängers**
- Identify **fake or impersonator profiles**
- Analyze identity similarity using:
  - Face
  - Name
  - Bio

---

## 🏗️ Project Structure
project_root/
│
├── app.py
├── utils/
│ ├── matcher.py
│ ├── enhancer.py
│ ├── insta_scraper.py
│
├── templates/
│ ├── index.html
│ ├── result.html
│ ├── search_instagram.html
│ ├── instagram_result.html
│
├── static/
│ ├── uploads/
│ ├── matches/
│ ├── annotated_uploads/
│ ├── insta_profiles/
│ ├── temp_enhanced/
│
├── faces_dataset/
├── instagram_cookies.json
└── README.md

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/doppelganger-detector.git
cd doppelganger-detector
2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows

3. Install dependencies
pip install -r requirements.txt

▶️ Run the Project
python app.py
Open in browser:

http://127.0.0.1:5000

📸 Usage
🔹 Face Matching
Upload image
System detects faces
Matches with dataset
Shows results with confidence
🔹 Instagram Search
Go to /search_instagram
Enter username (e.g., virat.kohli)
System:
Fetches profile
Downloads image
Runs matching pipeline

🔐 Instagram Login Setup

To enable deeper scraping:

Run hybrid login server
Login manually via browser
Save cookies to:
instagram_cookies.json


📊 Output
Match images
Confidence scores
CSV logs (static/match_log.csv)
Annotated faces

⚠️ Limitations
Instagram restricts:
Followers
Mutual friends
Only public profile data is reliably accessible
Dataset quality affects accuracy


🔮 Future Improvements
🔹 Face clustering (group similar people)
🔹 Database integration (SQLite / MongoDB)
🔹 Deepfake detection
🔹 Cross-platform profile search
🔹 Real-time webcam matching
🔹 AI-generated image verification

🧠 Tech Stack
Python
Flask
InsightFace
OpenCV
Instaloader
Selenium (for login handling)

👨‍💻 Contributors

Debmalya Panda
Aritra Paul
Sayan Hazra
Sayanti Roy

---
📜 License

This project is for academic and research purposes only.

# 📁 ✅ `.gitignore`

```gitignore
# ===============================
# Python
# ===============================
__pycache__/
*.pyc
*.pyo
*.pyd

# ===============================
# Virtual Environment
# ===============================
.venv/
venv/
.env/

# ===============================
# Chrome & Selenium
# ===============================
chrome-win64/
chromedriver-win64/
selenium_session/

# ===============================
# Logs
# ===============================
*.log

# ===============================
# OS Files
# ===============================
.DS_Store
Thumbs.db

# ===============================
# Dataset (large files)
# ===============================
faces_dataset/

# ===============================
# Temporary & Generated Files
# ===============================
static/temp_enhanced/
static/matches/
static/annotated_uploads/
static/insta_profiles/

# ===============================
# Secrets / Credentials
# ===============================
instagram_cookies.json

# ===============================
# IDE / Editor
# ===============================
.vscode/
.idea/
