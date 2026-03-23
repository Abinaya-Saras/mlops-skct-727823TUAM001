# MLOps Assignment – Diabetes Progression Prediction

**Name:** Abinaya Saras  
**Roll No:** 727823TUAM001  
**Dataset:** Diabetes Progression Prediction  

---

## 📌 Project Overview
This project predicts the progression of diabetes in patients using machine learning models. It follows MLOps practices including experiment tracking using MLflow and pipeline structuring.

---

## 🔄 Project Workflow

### 1. Data Loading
- The Diabetes dataset is loaded from scikit-learn.
- It contains 10 medical features such as age, BMI, and blood pressure.

### 2. Data Exploration (EDA)
- Performed using seaborn and matplotlib.
- Visualizations include:
  - Distribution plot
  - Correlation heatmap
  - Boxplot

### 3. Model Training
- Multiple models are trained:
  - Linear Regression
  - Random Forest Regressor
- 12 experiments are conducted with different configurations.

### 4. Experiment Tracking (MLflow)
- Each run logs:
  - Metrics: MAE, RMSE, R², MAPE
  - Parameters: model type, random seed
  - Tags: student name, roll number, dataset
- Best model is identified based on highest R² score.

### 5. Pipeline Design (Azure ML)
- Pipeline includes 3 stages:
  - Data Preparation
  - Model Training
  - Model Evaluation

### 6. Model Selection
- Random Forest performed better than Linear Regression.
- Best model is selected based on performance metrics.

---

## 📊 Results Summary
- Total runs: 12
- Best Model: Random Forest Regressor
- Best R² Score: ~0.7 (approx)

---

## ⚙️ Tools & Technologies
- Python  
- Scikit-learn  
- MLflow  
- Azure ML  
- Pandas, NumPy  
- Matplotlib, Seaborn  

---

## ▶️ How to Run

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run MLflow
mlflow ui

# Run training script
python code/training.py
