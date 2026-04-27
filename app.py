import streamlit as st

# MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Accident Severity Prediction", layout="wide")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ============================================
# LOAD SAVED MODEL AND FILES
# ============================================
@st.cache_resource
def load_artifacts():
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    le_weather = joblib.load('le_weather.pkl')
    le_road = joblib.load('le_road.pkl')
    le_light = joblib.load('le_light.pkl')
    le_vehicle = joblib.load('le_vehicle.pkl')
    feature_cols = joblib.load('feature_cols.pkl')
    severity_mapping = joblib.load('severity_mapping.pkl')
    reverse_mapping = {v: k for k, v in severity_mapping.items()}
    return model, scaler, le_weather, le_road, le_light, le_vehicle, feature_cols, reverse_mapping

@st.cache_data
def load_data():
    return pd.read_csv('unified_accident_data.csv')

# Load everything
model, scaler, le_weather, le_road, le_light, le_vehicle, feature_cols, reverse_mapping = load_artifacts()
df = load_data()

# Get available options
weather_options = list(le_weather.classes_)
road_options = list(le_road.classes_)
light_options = list(le_light.classes_)
vehicle_options = list(le_vehicle.classes_)

print("Weather classes:", weather_options)
print("Road classes:", road_options)
print("Light classes:", light_options)
print("Vehicle classes:", vehicle_options)
print("Features expected by model:", feature_cols)

# ============================================
# STREAMLIT UI
# ============================================
st.sidebar.title("🚗 Navigation")
option = st.sidebar.radio("Go to", ["🏠 Home", "📊 EDA", "🔮 Predict"])

# ============================================
# HOME PAGE
# ============================================
if option == "🏠 Home":
    st.title("🚗 Accident Severity Prediction")
    st.markdown("""
    This app predicts the severity of a road accident based on multiple factors.
    
    ### How it works:
    1. The model was trained on **UK accident records**
    2. Uses **12 features**: Weather, Road Type, Light, Vehicle Type, Speed, Vehicles, Casualties, Hour, Driver Age, and derived features
    3. Predicts severity on a scale of **1-3** (1=Least severe, 3=Most severe)
    
    ### Navigation:
    - **📊 EDA** - Explore data visualizations
    - **🔮 Predict** - Get severity predictions
    """)
    
    st.subheader("📋 Dataset Overview")
    st.write(f"Total records: {len(df):,}")
    st.dataframe(df.head(10))

# ============================================
# EDA PAGE
# ============================================
elif option == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")
    
    # Sample data for faster plotting
    df_sample = df.sample(n=min(10000, len(df)), random_state=42)
    
    # Create 2x2 grid of plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Severity Distribution
    severity_counts = df_sample['severity'].value_counts().sort_index()
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    severity_counts.plot(kind='bar', ax=axes[0,0], color=colors[:len(severity_counts)])
    axes[0,0].set_title('Accident Severity Distribution', fontsize=14)
    axes[0,0].set_xlabel('Severity (1=Least, 3=Most)', fontsize=12)
    axes[0,0].set_ylabel('Count', fontsize=12)
    
    # Plot 2: Weather vs Severity
    if 'weather' in df_sample.columns:
        weather_sev = pd.crosstab(df_sample['weather'], df_sample['severity'], normalize='index')
        weather_sev.plot(kind='bar', stacked=True, ax=axes[0,1], color=colors)
        axes[0,1].set_title('Weather Condition vs Severity', fontsize=14)
        axes[0,1].set_xlabel('Weather', fontsize=12)
        axes[0,1].set_ylabel('Proportion', fontsize=12)
        axes[0,1].legend(title='Severity')
        plt.setp(axes[0,1].get_xticklabels(), rotation=45, ha='right')
    else:
        axes[0,1].text(0.5, 0.5, 'Weather data not available', ha='center', va='center')
        axes[0,1].set_title('Weather Condition vs Severity')
    
    # Plot 3: Hour vs Severity
    if 'hour' in df_sample.columns:
        hour_severity = df_sample.groupby('hour')['severity'].mean()
        hour_severity.plot(kind='line', marker='o', ax=axes[1,0], color='blue', linewidth=2)
        axes[1,0].set_title('Mean Severity by Hour of Day', fontsize=14)
        axes[1,0].set_xlabel('Hour (0-23)', fontsize=12)
        axes[1,0].set_ylabel('Mean Severity', fontsize=12)
        axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Speed Limit vs Severity (if column exists)
    if 'Speed_limit' in df_sample.columns:
        df_sample['speed_bin'] = pd.cut(df_sample['Speed_limit'], bins=[0,30,50,70,100,200])
        speed_severity = df_sample.groupby('speed_bin')['severity'].mean()
        speed_severity.plot(kind='bar', ax=axes[1,1], color='green')
        axes[1,1].set_title('Speed Limit vs Severity', fontsize=14)
        axes[1,1].set_xlabel('Speed Limit (km/h)', fontsize=12)
        axes[1,1].set_ylabel('Mean Severity', fontsize=12)
        plt.setp(axes[1,1].get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Additional stats
    st.subheader("📈 Key Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Most Common Severity", f"{df['severity'].mode()[0]}")
    with col2:
        st.metric("Average Severity", f"{df['severity'].mean():.2f}")
    with col3:
        if 'weather' in df.columns:
            st.metric("Most Common Weather", f"{df['weather'].mode()[0]}")
        else:
            st.metric("Most Common Weather", "N/A")

# ============================================
# PREDICT PAGE
# ============================================
elif option == "🔮 Predict":
    st.title("🔮 Predict Accident Severity")
    st.markdown("Enter the details below to get a prediction:")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            weather = st.selectbox("🌤️ Weather Condition", options=weather_options)
            road = st.selectbox("🛣️ Road Type", options=road_options)
            light = st.selectbox("💡 Light Condition", options=light_options)
            vehicle = st.selectbox("🚗 Primary Vehicle Type", options=vehicle_options)
            speed_limit = st.number_input("🚗 Speed Limit (km/h)", min_value=0, max_value=200, value=50)
        
        with col2:
            num_vehicles = st.number_input("🚙 Number of Vehicles Involved", min_value=1, max_value=20, value=2)
            num_casualties = st.number_input("👥 Number of Casualties", min_value=0, max_value=50, value=0)
            hour = st.slider("⏰ Hour of Day", min_value=0, max_value=23, value=12)
            driver_age = st.number_input("👨 Average Driver Age", min_value=16, max_value=100, value=40)
        
        submitted = st.form_submit_button("🔮 Predict Severity", use_container_width=True)
    
    if submitted:
        try:
            # Encode categorical inputs
            weather_enc = le_weather.transform([weather])[0]
            road_enc = le_road.transform([road])[0]
            light_enc = le_light.transform([light])[0]
            vehicle_enc = le_vehicle.transform([vehicle])[0]
            
            # Calculate derived features
            casualty_per_vehicle = num_casualties / (num_vehicles + 1)
            is_night = 1 if (hour < 6 or hour > 20) else 0
            is_rush_hour = 1 if ((hour >= 7 and hour <= 9) or (hour >= 17 and hour <= 19)) else 0
            
            # Create input array with ALL 12 features in the correct order
            # Order must match feature_cols from training
            input_data = np.array([[
                weather_enc,      # weather_enc
                road_enc,         # road_enc
                light_enc,        # light_enc
                vehicle_enc,      # vehicle_enc
                speed_limit,      # Speed_limit
                num_vehicles,     # Number_of_Vehicles
                num_casualties,   # Number_of_Casualties
                hour,             # hour
                driver_age,       # avg_driver_age
                casualty_per_vehicle,  # casualty_per_vehicle
                is_night,         # is_night
                is_rush_hour      # is_rush_hour
            ]])
            
            # Get numerical feature indices for scaling
            num_feature_names = ['Speed_limit', 'Number_of_Vehicles', 'Number_of_Casualties', 
                                 'hour', 'avg_driver_age', 'casualty_per_vehicle']
            
            # Find indices of numerical features in the input array
            num_indices = []
            for i, col in enumerate(feature_cols):
                if col in num_feature_names:
                    num_indices.append(i)
            
            # Scale numerical features
            input_data[:, num_indices] = scaler.transform(input_data[:, num_indices])
            
            # Predict
            pred_remapped = model.predict(input_data)[0]
            pred_original = reverse_mapping[pred_remapped]
            proba = model.predict_proba(input_data)[0]
            
            # Display results
            st.success(f"## 🎯 Predicted Severity: **{pred_original}**")
            
            # Severity interpretation
            severity_labels = {
                1: "🟢 **Minor** - Low impact, minor damage",
                2: "🟡 **Serious** - Significant damage, possible injuries",
                3: "🔴 **Severe** - Major damage, serious injuries likely"
            }
            st.markdown(severity_labels.get(pred_original, "Unknown severity"))
            
            # Probability bars
            st.subheader("📊 Prediction Probabilities")
            proba_df = pd.DataFrame({
                'Severity': [reverse_mapping[i] for i in range(len(proba))],
                'Probability': proba
            })
            
            fig, ax = plt.subplots(figsize=(8, 4))
            colors_prob = ['#ff9999', '#66b3ff', '#99ff99']
            ax.barh(proba_df['Severity'], proba_df['Probability'], color=colors_prob[:len(proba_df)])
            ax.set_xlabel('Probability')
            ax.set_title('Probability Distribution by Severity')
            st.pyplot(fig)
            
            # Input summary
            with st.expander("📝 View Input Summary"):
                st.write(f"**Weather:** {weather}")
                st.write(f"**Road Type:** {road}")
                st.write(f"**Light Condition:** {light}")
                st.write(f"**Primary Vehicle Type:** {vehicle}")
                st.write(f"**Speed Limit:** {speed_limit} km/h")
                st.write(f"**Vehicles:** {num_vehicles}")
                st.write(f"**Casualties:** {num_casualties}")
                st.write(f"**Hour:** {hour}:00")
                st.write(f"**Average Driver Age:** {driver_age}")
                
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.write("Please check that all inputs are valid.")

# ============================================
# FOOTER
# ============================================
st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Model Info:**
    - Random Forest Classifier
    - 12 Features
    - SMOTE Balanced
    - Trained on UK accident records
    """
)