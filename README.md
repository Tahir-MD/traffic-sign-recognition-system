# 🚦 Traffic Sign Recognition System

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-FF4B4B.svg)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8.1-5C3EE8.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/yourusername/traffic-sign-detection/pulls)
[![Made with](https://img.shields.io/badge/Made%20with-❤️-red.svg)](https://github.com/yourusername/traffic-sign-detection)

## 📋 Overview

**Advanced Traffic Sign Recognition System** powered by YOLOv8 architecture. Designed for autonomous driving applications with real-time performance and industry-grade reliability. This system can detect and classify 43 different traffic signs with high accuracy and speed.

## ✨ Features

- 🚦 **Real-time Detection** - YOLOv8 based object detection
- 📸 **Image Processing** - Upload and detect signs in images
- 🎥 **Video Analysis** - Batch processing for videos
- 📹 **Live Webcam** - Real-time webcam detection with FPS counter
- 📊 **Performance Dashboard** - Live metrics and analytics
- 🎯 **43 Traffic Sign Classes** - Full GTSDB/GTSRB support
- ⚡ **Optimized Speed** - 45 FPS on GPU, 18 FPS on CPU
- 🐳 **Docker Ready** - Containerized deployment
- 🔧 **Customizable** - Adjust confidence and NMS thresholds
- 📈 **Training History** - View model performance over time

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip (latest version)
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/traffic-sign-detection.git
cd traffic-sign-detection

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py