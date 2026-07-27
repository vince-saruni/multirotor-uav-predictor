import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==========================================
# 1. PAGE CONFIGURATION & ANIMATED CSS UI
# ==========================================
st.set_page_config(
    page_title="UAV Energy Consumption & EDA Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Animated Background & Styled Components
st.markdown("""
<style>
    /* Animated Gradient Background */
    .stApp {
        background: linear-gradient(-45deg, #111184, #0047AB, #000080, #0F52BA, #007BA7);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        color: #f8fafc;
    }

    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Custom Metric Display */
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        color: #38bdf8;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. DATA LOADING & MODEL TRAINING (CACHED)
# ==========================================
@st.cache_data
def load_and_train_model():
    # Load dataset
    df = pd.read_csv('Comprehensive_Data_1_025454.csv')
    
    # Feature Selection & Target Identification
    features = [
        'Flight_Time_s', 'T_Climb_s', 'T_Hover_s', 'T_Dash_s', 'T_Loiter_s', 
        'T_Fig8_s', 'T_Land_s', 'Payload_kg', 'Wind_ms', 'Cruise_Alt_m', 'Avg_Speed_ms'
    ]
    target = 'Battery_Consumed_Pct'

    X = df[features]
    y = df[target]

    # Split and Train Model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBRegressor(
        n_estimators=120,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Calculate Evaluation Metrics
    y_pred = model.predict(X_test)
    metrics = {
        'r2': r2_score(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred)
    }

    return df, model, features, metrics

# Load cached data and trained model
try:
    df, model, feature_names, metrics = load_and_train_model()
except FileNotFoundError:
    st.error("Dataset file `Comprehensive_Data_1_025454.csv` not found. Please upload it to your workspace.")
    st.stop()


# ==========================================
# 3. NAVIGATION & SIDEBAR INPUT BOXES
# ==========================================
st.sidebar.title("UAV Control Panel")
app_mode = st.sidebar.radio("Navigate", ["Energy Prediction", "Exploratory Data Analysis (EDA)"])

st.sidebar.markdown("---")
st.sidebar.subheader("Flight & Mission Inputs")

# Sidebar input boxes (Numeric Text Inputs)
user_inputs = {}
for col in feature_names:
    min_val = float(df[col].min())
    max_val = float(df[col].max())
    mean_val = float(df[col].mean())
    
    # Determine step size based on float or int
    step = 0.1 if isinstance(df[col].iloc[0], (float, np.floating)) else 1.0
    
    user_inputs[col] = st.sidebar.number_input(
        label=f"{col.replace('_', ' ')}",
        min_value=min_val,
        max_value=max_val,
        value=round(mean_val, 2),
        step=step,
        help=f"Valid range: {min_val} to {max_val}"
    )


# ==========================================
# 4. VIEW 1: PREDICTION DASHBOARD
# ==========================================
if app_mode == "Energy Prediction":
    st.title("Multirotor UAV Battery Consumption Predictor")
    st.markdown("Enter flight parameters in the input boxes on the sidebar to estimate battery consumption.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Model Performance Metrics")
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("R² Accuracy Score", f"{metrics['r2'] * 100:.2f}%")
        m_col2.metric("RMSE", f"{metrics['rmse']:.3f}%")
        m_col3.metric("MAE", f"{metrics['mae']:.3f}%")

        st.markdown("---")
        st.subheader("Run Prediction")
        
        # Trigger prediction button
        if st.button("Calculate Battery Consumption", type="primary", use_container_width=True):
            input_df = pd.DataFrame([user_inputs])
            prediction = model.predict(input_df)[0]

            if (100.0 - prediction) >= 30:
                st.success("Calculation Complete! The Mission is feasible")
            elif (100.0 - prediction) <= 30:
                st.success("Calculation Complete! The Mission is NOT feasible")
            
            p_col1, p_col2 = st.columns(2)
            p_col1.metric(
                label="Predicted Battery Consumption", 
                value=f"{prediction:.2f}%"
            )
            p_col2.metric(
                label="Estimated Remaining SOC", 
                value=f"{max(0.0, 100.0 - prediction):.2f}%"
            )
            
            # Progress bar visualizer
            st.write("Battery Consumption Indicator:")
            st.progress(min(max(float(prediction) / 100.0, 0.0), 1.0))

    with col2:
        st.subheader("Active Mission Configuration")
        input_summary = pd.DataFrame(list(user_inputs.items()), columns=["Parameter", "Entered Value"])
        st.dataframe(input_summary, use_container_width=True, height=400)


# ==========================================
# 5. VIEW 2: EDA DASHBOARD
# ==========================================
else:
    st.title("Exploratory Data Analysis (EDA)")
    st.markdown("Explore dataset distributions, feature correlations, and spread across variables.")

    eda_tab1, eda_tab2, eda_tab3 = st.tabs(["Correlation Heatmap", "Histograms", "Boxplots"])

    # Seaborn Style Configuration for Dark Background
    plt.style.use('dark_background')

    # Tab 1: Correlation Heatmap
    with eda_tab1:
        st.subheader("Feature Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        
        corr = df.corr(numeric_only=True)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="mako", ax=ax, cbar=True, linewidths=0.5)
        st.pyplot(fig)

    # Tab 2: Histograms
    with eda_tab2:
        st.subheader("Feature Distribution Histograms")
        selected_hist_col = st.selectbox("Select Feature for Histogram", df.columns, index=0)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        
        sns.histplot(df[selected_hist_col], kde=True, color="#38bdf8", bins=30, ax=ax)
        ax.set_title(f"Distribution of {selected_hist_col}", color="white")
        st.pyplot(fig)

    # Tab 3: Boxplots
    with eda_tab3:
        st.subheader("Feature Outlier Analysis (Boxplots)")
        selected_box_col = st.selectbox("Select Feature for Boxplot", df.columns, index=7)
        
        fig, ax = plt.subplots(figsize=(8, 3))
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        
        sns.boxplot(x=df[selected_box_col], color="#a855f7", ax=ax)
        ax.set_title(f"Boxplot of {selected_box_col}", color="white")
        st.pyplot(fig)
