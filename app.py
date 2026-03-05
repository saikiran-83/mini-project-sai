import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Page Configuration
st.set_page_config(page_title="Doctor Visits Analysis & Prediction", layout="wide")


def inject_css():
    css = '''
    <style>
    /* Use the default page background but provide distinct foreground styling */
    body, .stApp { background: inherit !important; }

    /* Gradient, clipped title for a unique look */
    .stApp h1, .stApp .css-1v3fvcr h1 {
        font-weight: 800;
        font-size: 2.2rem;
        background: linear-gradient(90deg, #ff6b6b, #ffb86b, #6bdeff);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        margin-bottom: 6px;
    }

    /* Contrasting block container with left accent */
    .stApp .block-container {
        background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(250,250,255,0.98)) !important;
        border-radius: 14px;
        padding: 18px;
        border-left: 6px solid #6bdeff;
        box-shadow: 0 10px 30px rgba(11,86,118,0.06);
    }

    /* Sidebar: colorful and distinct */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b5676, #114e76) !important;
        color: #ffffff !important;
        padding-top: 18px !important;
    }
    [data-testid="stSidebar"] .css-1d391kg, [data-testid="stSidebar"] .stText {
        color: #ffffff !important;
    }

    /* Buttons with vibrant gradient */
    .stButton>button {
        background: linear-gradient(90deg,#ff6b6b,#6bdeff) !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 6px 18px rgba(107, 222, 255, 0.12) !important;
        border-radius: 10px !important;
    }

    /* Small table/panel tweaks for clarity */
    .stTable td, .stTable th { padding: 8px 12px; }

    /* Ensure plot backgrounds are transparent to fit design */
    .plotly-graph-div .main-svg { background: transparent !important; }

    </style>
    '''
    st.markdown(css, unsafe_allow_html=True)

inject_css()
st.title("🏥 Doctor Visits Analysis & Prediction App")
st.markdown("""
This application showcases an end-to-end data analysis pipeline:
1. **Exploratory Data Analysis (EDA)**: Understanding the dataset structure.
2. **Visualization**: Visual insights into patient demographics and health.
3. **Machine Learning**: A Random Forest model to predict the number of doctor visits based on patient attributes.
""")

# --- 1. Load Data ---
@st.cache_data
def load_data():
    # Make sure the filename matches exactly what you have
    file_path = "1719219834-DoctorVisits - DA (1) (1).csv"
    try:
        df = pd.read_csv(file_path)
        # Drop the index column if it exists
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        return df
    except FileNotFoundError:
        st.error(f"File not found: {file_path}. Please make sure the CSV file is in the same directory.")
        return None


df = load_data()

if df is not None:
    # --- Preprocessing Function ---
    def preprocess_data(data):
        # Create a copy to avoid SettingWithCopyWarning
        df_processed = data.copy()
        
        # Mapping Binary Categorical Columns
        binary_cols = ['private', 'freepoor', 'freerepat', 'nchronic', 'lchronic']
        for col in binary_cols:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].map({'yes': 1, 'no': 0})
        
        # Mapping Gender
        if 'gender' in df_processed.columns:
            df_processed['gender'] = df_processed['gender'].map({'female': 0, 'male': 1})
            
        return df_processed

    # Apply preprocessing
    df_clean = preprocess_data(df)

    # --- Tabs Setup ---
    tab1, tab2, tab3 = st.tabs(["📊 Exploratory Data Analysis", "📈 Data Visualization", "🤖 Machine Learning Model"])

    # --- TAB 1: EDA ---
    with tab1:
        st.header("Exploratory Data Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Snapshot")
            st.dataframe(df.head(10))
            
        with col2:
            st.subheader("Dataset Statistics")
            st.write(df.describe())
            
        st.subheader("Data Information")
        buffer = pd.DataFrame(df.dtypes, columns=['Data Type'])
        buffer['Null Values'] = df.isnull().sum()
        # Convert dtype objects to strings to avoid Arrow serialization issues in Streamlit
        buffer['Data Type'] = buffer['Data Type'].astype(str)
        st.table(buffer)

    # --- TAB 2: Visualization ---
    with tab2:
        st.header("Data Visualization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribution of Doctor Visits")
            fig1, ax1 = plt.subplots()
            sns.histplot(df['visits'], bins=10, kde=False, color='skyblue', ax=ax1)
            ax1.set_title("Histogram of Visits")
            st.pyplot(fig1)
            
            st.subheader("Visits by Gender")
            fig2, ax2 = plt.subplots()
            sns.barplot(x='gender', y='visits', data=df, palette='pastel', ax=ax2)
            ax2.set_title("Average Visits by Gender")
            st.pyplot(fig2)

        with col2:
            st.subheader("Income Distribution")
            fig3, ax3 = plt.subplots()
            sns.histplot(df['income'], kde=True, color='green', ax=ax3)
            ax3.set_title("Income Distribution")
            st.pyplot(fig3)
            
            st.subheader("Correlation Heatmap")
            # Compute correlation on processed (numeric) data
            corr = df_clean.corr()
            fig4, ax4 = plt.subplots()
            sns.heatmap(corr, annot=True, fmt=".1f", cmap='coolwarm', ax=ax4, annot_kws={"size": 8})
            st.pyplot(fig4)

        st.subheader("Illness vs Visits")
        fig5, ax5 = plt.subplots(figsize=(10, 4))
        sns.boxplot(x='illness', y='visits', data=df, palette='Set2', ax=ax5)
        ax5.set_title("Doctor Visits based on Number of Illnesses")
        st.pyplot(fig5)

    # --- TAB 3: Machine Learning ---
    with tab3:
        st.header("Predict Doctor Visits")
        st.write("We use a **Random Forest Regressor** to predict the number of doctor visits based on patient details.")
        
        # Prepare Data
        X = df_clean.drop(columns=['visits'])
        y = df_clean['visits']
        
        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Model Training
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Predictions & Metrics
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        st.subheader("Model Performance")
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Mean Squared Error (MSE)", f"{mse:.4f}")
        col_m2.metric("R² Score", f"{r2:.4f}")
        
        st.markdown("---")
        st.subheader("🔮 Live Prediction")
        st.write("Enter patient details below to predict expected doctor visits:")
        
        # User Input Form
        input_col1, input_col2, input_col3 = st.columns(3)
        
        with input_col1:
            gender = st.selectbox("Gender", ["female", "male"])
            age = st.slider("Age (Normalized)", 0.0, 1.0, 0.2)
            income = st.slider("Income (Normalized)", 0.0, 1.0, 0.5)
            illness = st.number_input("Number of Illnesses", 0, 5, 1)

        with input_col2:
            reduced = st.number_input("Reduced Activity Days", 0, 14, 0)
            health = st.number_input("Health Score", 0, 12, 1)
            private = st.selectbox("Private Insurance", ["yes", "no"])
            freepoor = st.selectbox("Free Poor", ["yes", "no"])

        with input_col3:
            freerepat = st.selectbox("Free Repat", ["yes", "no"])
            nchronic = st.selectbox("Non-Chronic Condition", ["yes", "no"])
            lchronic = st.selectbox("Long-Chronic Condition", ["yes", "no"])
        
        # Predict Button
        if st.button("Predict Visits"):
            # Create a dataframe from input
            input_data = pd.DataFrame({
                'gender': [gender],
                'age': [age],
                'income': [income],
                'illness': [illness],
                'reduced': [reduced],
                'health': [health],
                'private': [private],
                'freepoor': [freepoor],
                'freerepat': [freerepat],
                'nchronic': [nchronic],
                'lchronic': [lchronic]
            })
            
            # Preprocess input
            input_processed = preprocess_data(input_data)
            
            # Ensure columns match training data order
            input_processed = input_processed[X.columns]
            
            # Prediction
            prediction = model.predict(input_processed)[0]
            
            st.success(f"Estimated Number of Doctor Visits: **{prediction:.2f}**")
            
            if prediction < 1:
                st.info("The model predicts this patient is unlikely to visit the doctor.")
            else:
                st.warning("The model predicts a high likelihood of doctor visits.")
