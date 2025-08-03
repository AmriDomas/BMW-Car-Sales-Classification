# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# ========== Load ==========
@st.cache_data
def load_data():
    return pd.read_csv("BMW_Car_Sales_Classification.csv")  # data mentah

@st.cache_resource
def load_model():
    with open("best_xgb_model.pkl", "rb") as f:
        return pickle.load(f)

# ========== Feature Engineering ==========
def preprocess_input(df_raw, model):
    car = df_raw.copy()

    # --- Feature Engineering ---
    car['car_age'] = 2025 - car['Year']
    car['engine_size_per_year'] = car['Engine_Size_L'] / car['car_age']
    car['mileage_per_year'] = car['Mileage_KM'] / car['car_age']

    car['Price_Category'] = pd.cut(
        car['Price_USD'],
        bins=[0, 50000, 90000, np.inf],
        labels=["Cheap", "Medium", "Expensive"]
    )

    luxury_models = ['7 Series', '8 Series', 'i8', 'M Series']
    car['Model_Category'] = car['Model'].apply(lambda x: 'Luxury' if x in luxury_models else 'Standard')

    top_colors = ['Black', 'White', 'Grey']  # disesuaikan dari top 3 waktu training
    car['Color_Category'] = car['Color'].apply(lambda x: 'Popular' if x in top_colors else 'Rare')

    car['Efficiency_Index'] = car['Mileage_KM'] / car['Engine_Size_L']
    car['Price_per_KM'] = car['Price_USD'] / car['Mileage_KM'].replace(0, np.nan)
    car['Log_Price_per_KM'] = np.log(car['Price_per_KM'].replace(0, np.nan)).replace([-np.inf, np.inf], np.nan).fillna(0)

    # --- Encode kategorikal ---
    car = pd.get_dummies(car, drop_first=True)

    # --- Tambah kolom yang hilang & pastikan urutan sesuai model ---
    model_features = model.feature_names_in_
    for col in model_features:
        if col not in car.columns:
            car[col] = 0  # Kolom yang nggak muncul di input user → default 0

    # --- Urutkan kolom agar match dengan model ---
    car = car[model_features]

    return car

# ========== App ==========
st.set_page_config(layout="wide")
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Select Menu", ["Analyst", "Prediction"])

# ANALYST PANEL
if menu == "Analyst":
    st.title("🔎 Analyst Panel")
    df = load_data()
    st.dataframe(df.head())
    st.write("Basic Stats:")
    st.write(df.describe())

    st.subheader("Distribusi Kategorikal")
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    for col in cat_cols:
        st.markdown(f"**{col}**")
        st.bar_chart(df[col].value_counts())
    
    # --- Korelasi Numerik (Heatmap) dengan Font Proporsional ---
    st.subheader("🔗 Korelasi Numerik (Heatmap)")
    car_numeric = df.select_dtypes(include=['number'])
    corr_matrix = car_numeric.corr()

    fig1, ax1 = plt.subplots(figsize=(7, 5))
    heatmap = sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        annot_kws={"size": 5},   # Ukuran angka dalam sel
        cbar_kws={"shrink": 0.5} # Skala warna lebih ramping
    )
    ax1.set_title("Heatmap Correlation", fontsize=10)
    ax1.tick_params(axis='x', labelsize=6, rotation=90)
    ax1.tick_params(axis='y', labelsize=6, rotation=0)

    # Perkecil tulisan colorbar (legend)
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=6)

    st.pyplot(fig1)

    # --- Kategorikal Countplot dengan Auto Orientation ---
    st.subheader("📊 Distribusi Variabel Kategorikal")
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    cat_cols = [col for col in cat_cols if col != 'Sales_Classification']

    selected_col = st.selectbox("Pilih kolom kategorikal untuk visualisasi", cat_cols)

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    num_categories = df[selected_col].nunique()

    if num_categories > 10:
        # Horizontal barplot
        sns.countplot(y=selected_col, hue='Sales_Classification', data=df, ax=ax2)
        ax2.set_title(f'{selected_col} by Sales Classification', fontsize=10)
        ax2.set_xlabel('Count', fontsize=9)
        ax2.set_ylabel(selected_col, fontsize=9)
        ax2.tick_params(axis='y', labelsize=8)
        ax2.tick_params(axis='x', labelsize=8)
    else:
        # Vertical barplot
        sns.countplot(x=selected_col, hue='Sales_Classification', data=df, ax=ax2)
        ax2.set_title(f'{selected_col} by Sales Classification', fontsize=10)
        ax2.set_xlabel(selected_col, fontsize=9)
        ax2.set_ylabel('Count', fontsize=9)
        ax2.tick_params(axis='x', rotation=90, labelsize=8)
        ax2.tick_params(axis='y', labelsize=8)

    # Styling judul & legend
    ax2.set_title(f'{selected_col} by Sales Classification', fontsize=7)
    legend = ax2.get_legend()
    if legend:
        legend.set_title("Sales Classification")
        for text in legend.get_texts():
            text.set_fontsize(3)
        legend.get_title().set_fontsize(4)

    # Label bar (fungsi ini akan otomatis menyesuaikan orientasi)
    for container in ax2.containers:
        ax2.bar_label(container, fmt='%d', label_type='edge', fontsize=5)

    st.pyplot(fig2)
    
    # --- Boxplot Numeric vs Sales Classification ---
    st.subheader("📦 Boxplot of Numeric Variables based on Sales Classification")

    # Pilih kolom numerik
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    selected_num = st.selectbox("Select numeric variables", num_cols, index=num_cols.index('Mileage_KM') if 'Mileage_KM' in num_cols else 0)

    fig3, ax3 = plt.subplots(figsize=(6, 4))
    sns.boxplot(x='Sales_Classification', y=selected_num, data=df, ax=ax3)

    # Styling axis & judul
    ax3.set_title(f'{selected_num} vs Sales Classification', fontsize=8)
    ax3.set_xlabel('Sales Classification', fontsize=5)
    ax3.set_ylabel(selected_num.replace('_', ' '), fontsize=5)
    ax3.tick_params(axis='both', labelsize=8)

    # Hitung median tiap kelas dan tampilkan di tengah box
    medians = df.groupby('Sales_Classification')[selected_num].median().reset_index()

    for i, row in medians.iterrows():
        x_coord = ax3.get_xticks()[i]
        median_val = row[selected_num]
        ax3.text(
            x_coord,
            median_val,
            f'{median_val:,.0f}',
            horizontalalignment='center',
            size=7,
            color='red',
            weight='semibold'
        )

    st.pyplot(fig3)


# PREDICTION PANEL
elif menu == "Prediction":
    st.title("🚀 Sales Classification Prediction")
    model = load_model()

    # Input Bar
    st.markdown("Masukkan fitur mobil (mentah):")
    model_ = st.selectbox("Model", ['1 Series', '3 Series', '5 Series', '7 Series', '8 Series', 'M Series', 'X1', 'X3', 'X5', 'i8'])
    year = st.number_input("Year", min_value=2000, max_value=2025, value=2010)
    region = st.selectbox("Region", ['US', 'Europe', 'Asia', 'Middle East'])
    color = st.selectbox("Color", ['Black', 'White', 'Blue', 'Red', 'Silver', 'Grey', 'Other'])
    fuel = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'Hybrid', 'Electric'])
    trans = st.selectbox("Transmission", ['Manual', 'Automatic'])
    engine = st.number_input("Engine Size (L)", min_value=0.5, max_value=8.0, value=1.5, step=0.1)
    mileage = st.number_input("Mileage (KM)", min_value=0, max_value=500000, value=3)
    price = st.number_input("Price (USD)", min_value=500, max_value=200000, value=30000)
    volume = st.number_input("Sales Volume", min_value=0, max_value=1000000, value=100)


    # DataFrame dummy input
    input_df = pd.DataFrame([{
        'Model': model_,
        'Year': year,
        'Region': region,
        'Color': color,
        'Fuel_Type': fuel,
        'Transmission': trans,
        'Engine_Size_L': engine,
        'Mileage_KM': mileage,
        'Price_USD': price,
        'Sales_Volume': volume,
    }])

    if st.button("Predict"):
       processed_input = preprocess_input(input_df, model)
       prediction = model.predict(processed_input)
       st.success(f"📊 Predicted Sales Classification: **{prediction[0]}**")
