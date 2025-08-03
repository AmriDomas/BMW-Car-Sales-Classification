# 🚗 BMW Car Sales Classification App (Streamlit)

A Streamlit-based interactive web app that classifies BMW car sales performance (`High` vs `Low`) using user-input specs — plus includes interactive data visualizations for analysis.

---

## 🧠 Project Overview

This app is part of a portfolio project focused on car sales data. The main goal is to:

- Predict whether a BMW car listing is likely to have **High or Low Sales Classification**
- Provide easy-to-use **interactive analysis tools** to explore the data and understand what drives sales

---

## 🧩 Dataset Features

The app is built on a dataset with 50,000 BMW car records and includes the following raw features:

- `Model`, `Year`, `Region`, `Color`, `Fuel_Type`, `Transmission`
- `Engine_Size_L`, `Mileage_KM`, `Price_USD`, `Sales_Volume`
- Target: `Sales_Classification` (High / Low)

The model was trained with feature engineering (car age, efficiency index, price categories, etc), but user input in the app uses only **raw/original columns**.

---

## 🧭 Navigation

The app is split into two panels:

### 🔍 Analyst Panel

Includes:

- 📊 **Dynamic Count Plot**  
  Select any categorical variable (e.g. `Model`, `Region`) to compare its sales classification distribution.  
  Bar orientation switches automatically if there are too many categories.

- 🔥 **Correlation Heatmap**  
  Shows relationships between all numerical features using a color-coded heatmap.

- 📦 **Interactive Boxplot**  
  Choose any numeric column (e.g. `Mileage_KM`, `Price_USD`) to compare distributions across `Sales_Classification`.  
  Annotated with median values per class.

---

### 📈 Prediction Panel

A user-friendly form that allows you to:

- Input car specs manually using dropdowns and number fields
- Submit and get a **prediction**: whether it will have High or Low Sales
- The model is pre-trained and loaded via `.pkl` file

---

## 🚀 How to Run It Locally

1. **Clone the repo**:
   ```bash
   git clone https://github.com/AmriDomas/BMW-Car-Sales-Classification.git 
   cd BMW-Car-Sales-Classification
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run streamlit_bmw.py
   ```

## 📦 Tech Stack

   - Python (pandas, scikit-learn, XGBoost, matplotlib, seaborn)
   - Streamlit (frontend UI)
   - Pickle model loading
   - Jupyter Notebook for data prep & training

## 📌 Notes

  - Input form uses only original features (before feature engineering), to make it user-friendly and clean.
  - Trained model includes extensive feature engineering behind the scenes.
  - All visualizations auto-scale for cleaner layout and better readability in both dark/light themes.

