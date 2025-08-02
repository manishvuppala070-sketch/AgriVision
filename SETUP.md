# Setup Guide

## Prerequisites
- Python 3.8 or higher
- Git installed on your system

## Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/AgriVision.git
   cd AgriVision
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download EuroSAT Dataset**
   - Download from: https://github.com/phelber/EuroSAT
   - Extract to `EuroSAT/` folder in project root
   - Ensure folder structure: `EuroSAT/AnnualCrop/`, `EuroSAT/Forest/`, etc.

5. **Create model directory**
   ```bash
   mkdir model
   ```

6. **Run the application**
   ```bash
   python Main.py
   ```

## Dataset Structure
```
EuroSAT/
├── AnnualCrop/
├── Forest/
├── HerbaceousVegetation/
├── Highway/
├── Industrial/
├── Pasture/
├── PermanentCrop/
├── Residential/
├── River/
└── SeaLake/
```