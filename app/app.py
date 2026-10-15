import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configurations (native Streamlit theme with wide layout)
st.set_page_config(
    page_title="AeroFlow AI: Airline Satisfaction Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Path configurations
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
model_path = os.path.join(project_dir, "models", "passenger_satisfaction_model.pkl")
scaler_path = os.path.join(project_dir, "models", "scaler.pkl")
data_path = os.path.join(project_dir, "data", "airline_passenger_satisfaction.csv")
css_path = os.path.join(current_dir, "style.css")

# Load Custom Glassmorphism CSS if available
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Define Default Values (Empty/Unselected states for presentation environment)
DEFAULT_VALUES = {
    'gender': 'Select Gender',
    'customer_type': 'Select Loyalty Type',
    'age': None,
    'travel_type': 'Select Travel Purpose',
    'flight_class': 'Select Class',
    'distance': None,
    'dep_delay': None,
    'arr_delay': None,
    'wifi': 'Select Rating',
    'booking': 'Select Rating',
    'boarding': 'Select Rating',
    'comfort': 'Select Rating',
    'ent': 'Select Rating',
    'clean': 'Select Rating',
    'onboard': 'Select Rating'
}

# Initialize session state keys if not already present
for key, val in DEFAULT_VALUES.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Callback function to reset all inputs to their unselected placeholder states
def reset_all():
    for key, val in DEFAULT_VALUES.items():
        st.session_state[key] = val

# Load model and scaler with caching
@st.cache_resource
def load_ml_assets():
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    return None, None

model, scaler = load_ml_assets()

# Load dataset for EDA with caching
@st.cache_data
def load_data():
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None

df = load_data()

# Header Banner
st.markdown("""
<div class="header-container">
    <div class="header-title">✈️ AeroFlow AI: Passenger Experience Portal</div>
    <div class="header-subtitle">REAL-TIME AIRLINE PASSENGER SATISFACTION CLASSIFIER & ANALYTICS DASHBOARD</div>
</div>
""", unsafe_allow_html=True)

if model is None or scaler is None:
    st.error("Error: Model or Scaler file not found. Please make sure to run the training script first!")
    st.stop()

# Set up tabs
tab1, tab2, tab3 = st.tabs(["🔮 Predict Satisfaction", "📊 Experience Analytics Dashboard", "🧠 Model Insights"])

# ----------------- TAB 1: PREDICT SATISFACTION -----------------
with tab1:
    st.markdown('<div class="section-header">Passenger Demographics & Flight Profile</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.write("### 👤 Passenger Profile")
        gender = st.selectbox("Gender", ["Select Gender", "Male", "Female"], key="gender", help="Gender of the passenger (UI only)")
        customer_type = st.selectbox("Customer Type", ["Select Loyalty Type", "Loyal Customer", "disloyal Customer"], key="customer_type", help="Is this passenger a repeat customer? (UI only)")
        age = st.number_input("Passenger Age (Years)", min_value=7, max_value=120, value=None, placeholder="Enter Passenger Age", key="age")
        travel_type = st.selectbox("Type of Travel", ["Select Travel Purpose", "Business travel", "Personal Travel"], key="travel_type", help="Purpose of flight (UI only)")
        flight_class = st.selectbox("Class", ["Select Class", "Business", "Eco", "Eco Plus"], key="flight_class", help="Flight class of travel (Model Feature)")
        distance = st.number_input("Flight Distance (miles)", min_value=10, max_value=10000, value=None, placeholder="Enter Flight Distance", key="distance", help="Flight route distance in miles (Model Feature)")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
        st.write("### ⏱️ Flight Status")
        dep_delay = st.number_input("Departure Delay in Minutes", min_value=0, max_value=1440, value=None, placeholder="Enter Departure Delay", key="dep_delay", help="Departure delay in minutes (UI only)")
        arr_delay = st.number_input("Arrival Delay in Minutes", min_value=0, max_value=1440, value=None, placeholder="Enter Arrival Delay", key="arr_delay", help="Arrival delay in minutes (Model Feature)")
        st.info("Flight delays directly affect passenger trust and rating scores. Enter actual delayed times in minutes.")
        st.markdown('</div>', unsafe_allow_html=True)
        
    st.markdown('<div class="section-header">Passenger Ratings of Inflight & Ground Services (Top Drivers)</div>', unsafe_allow_html=True)
    
    col_a, col_b = st.columns(2)
    rating_options = ["Select Rating", "0", "1", "2", "3", "4", "5"]
    
    with col_a:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.write("### 📶 Digital Experience Features")
        wifi = st.selectbox("Inflight Wifi Service", rating_options, key="wifi")
        booking = st.selectbox("Ease of Online Booking", rating_options, key="booking")
        boarding = st.selectbox("Online Boarding", rating_options, key="boarding")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_b:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.write("### 💺 In-Cabin Experience Features")
        comfort = st.selectbox("Seat Comfort", rating_options, key="comfort")
        ent = st.selectbox("Inflight Entertainment", rating_options, key="ent")
        clean = st.selectbox("Cleanliness", rating_options, key="clean")
        onboard = st.selectbox("On-board Service", rating_options, key="onboard")
        st.markdown('</div>', unsafe_allow_html=True)
        
    st.markdown("---")
    
    # Action buttons (Predict and Reset side-by-side)
    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        predict_clicked = st.button("✈️ Predict Passenger Satisfaction", use_container_width=True, type="primary")
    with col_btn2:
        st.button("🔄 Reset / Clear Form", on_click=reset_all, use_container_width=True)
        
    if predict_clicked:
        # Check if any field is unselected
        unselected = []
        if gender == "Select Gender": unselected.append("Gender")
        if customer_type == "Select Loyalty Type": unselected.append("Customer Type")
        if age is None: unselected.append("Passenger Age")
        if travel_type == "Select Travel Purpose": unselected.append("Type of Travel")
        if flight_class == "Select Class": unselected.append("Class")
        if distance is None: unselected.append("Flight Distance")
        if dep_delay is None: unselected.append("Departure Delay")
        if arr_delay is None: unselected.append("Arrival Delay")
        if wifi == "Select Rating": unselected.append("Inflight Wifi Service")
        if booking == "Select Rating": unselected.append("Ease of Online Booking")
        if boarding == "Select Rating": unselected.append("Online Boarding")
        if comfort == "Select Rating": unselected.append("Seat Comfort")
        if ent == "Select Rating": unselected.append("Inflight Entertainment")
        if clean == "Select Rating": unselected.append("Cleanliness")
        if onboard == "Select Rating": unselected.append("On-board Service")
        
        if unselected:
            st.error(f"⚠️ **Please fill all inputs before running predictions:** {', '.join(unselected)}")
        else:
            # Map values to internal numeric representations
            class_map = {'Business': 0, 'Eco': 1, 'Eco Plus': 2}
            
            # Convert selectbox rating strings to integers
            val_wifi = int(wifi)
            val_booking = int(booking)
            val_boarding = int(boarding)
            val_comfort = int(comfort)
            val_ent = int(ent)
            val_clean = int(clean)
            val_onboard = int(onboard)
            
            # Prepare inputs strictly in the order expected by the 10-feature model
            input_data = [
                val_boarding,
                val_clean,
                val_comfort,
                val_ent,
                val_wifi,
                val_booking,
                int(distance),
                val_onboard,
                class_map[flight_class],
                int(arr_delay)
            ]
            
            # Scale inputs
            input_df = pd.DataFrame([input_data])
            scaled_input = scaler.transform(input_df)
            
            # Predict Class and Probability
            prediction = model.predict(scaled_input)[0]
            prob = model.predict_proba(scaled_input)[0]
            
            # Display Prediction Result natively
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.write("## 🎫 Passenger Prediction Result")
            
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                st.write(f"**Passenger Details:** {gender} ({int(age)} Yrs)")
                st.write(f"**Cabin Class:** {flight_class}")
                st.write(f"**Loyalty Type:** {customer_type}")
                st.write(f"**Travel Reason:** {travel_type}")
                st.write(f"**Flight Distance:** {int(distance)} miles")
                st.write(f"**Arrival Delay:** {int(arr_delay)} minutes")
                st.write(f"**Flight Number:** AP-{int(age)}{int(distance)%100:02d}")
                
            with col_res2:
                if prediction == 1:
                    confidence = prob[1] * 100
                    st.success(f"### SATISFIED (🟢)\n**Confidence:** {confidence:.2f}%")
                    st.markdown("**Actionable Feedback:** Passenger experience meets overall quality expectations! Keep up the premium check-in and onboard service.")
                else:
                    confidence = prob[0] * 100
                    st.error(f"### DISSATISFIED / NEUTRAL (🔴)\n**Confidence:** {confidence:.2f}%")
                    
                    # Actionable advice based on negative ratings
                    low_ratings = []
                    if val_wifi < 3: low_ratings.append("Wifi Service (📶)")
                    if val_boarding < 3: low_ratings.append("Online Boarding (🎫)")
                    if val_comfort < 3: low_ratings.append("Seat Comfort (💺)")
                    if val_booking < 3: low_ratings.append("Online Booking (💻)")
                    if int(arr_delay) > 30: low_ratings.append("Flight Arrival Delays (⏱️)")
                    
                    if low_ratings:
                        st.warning(f"⚠️ **Key Churn Risk Factors:** Low ratings detected in **{', '.join(low_ratings)}**. Focus on resolving these core experiences to convert this passenger.")
                    else:
                        st.info("💡 **Retention tip:** Ratings are acceptable, but parameters suggest high expectations. Consider offering lounge upgrades or service coupons to secure satisfaction.")
            st.markdown('</div>', unsafe_allow_html=True)

# ----------------- TAB 2: DYNAMIC ANALYTICS -----------------
with tab2:
    if df is not None:
        st.markdown('<div class="section-header">Live Experience Dashboard</div>', unsafe_allow_html=True)
        
        # KPI metrics display using native Streamlit metric columns
        total_rows = len(df)
        sat_pct = (df['satisfaction'].value_counts(normalize=True)['satisfied']) * 100
        avg_wifi = df['Inflight_wifi_service'].mean()
        avg_dist = df['Flight_Distance'].mean()
        
        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
        col_kpi1.metric("TOTAL PASSENGERS ANALYZED", f"{total_rows:,}")
        col_kpi2.metric("SATISFACTION RATE", f"{sat_pct:.1f}%")
        col_kpi3.metric("AVG WIFI EXPERIENCE", f"{avg_wifi:.2f}/5")
        col_kpi4.metric("AVG FLIGHT ROUTE", f"{avg_dist:.0f} mi")
        
        st.markdown("---")
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.write("### 📊 Satisfaction Split by Flight Class")
            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor('none')
            ax.set_facecolor('none')
            sns.countplot(data=df, x='Class', hue='satisfaction', palette=['#f87171', '#34d399'], ax=ax)
            ax.set_title('Satisfaction Distribution by Travel Class', color='#f8fafc')
            ax.tick_params(colors='#94a3b8')
            ax.xaxis.label.set_color('#94a3b8')
            ax.yaxis.label.set_color('#94a3b8')
            st.pyplot(fig)
            plt.close()
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_chart2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.write("### 🔍 Interactive Service Impact Driver")
            rating_cols = [
                "Online_boarding", "Cleanliness", "Seat_comfort", "Inflight_entertainment", 
                "Inflight_wifi_service", "Ease_of_Online_booking", "On-board_service"
            ]
            selected_feature = st.selectbox("Select Service Rating to Visualize", rating_cols)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor('none')
            ax.set_facecolor('none')
            sns.countplot(data=df, x=selected_feature, hue='satisfaction', palette=['#f87171', '#34d399'], ax=ax)
            ax.set_title(f'Satisfaction Drivers vs {selected_feature.replace("_", " ").title()}', color='#f8fafc')
            ax.tick_params(colors='#94a3b8')
            ax.xaxis.label.set_color('#94a3b8')
            ax.yaxis.label.set_color('#94a3b8')
            st.pyplot(fig)
            plt.close()
            st.markdown('</div>', unsafe_allow_html=True)
            
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.write("### 📈 Passenger Age Profile Density")
        fig, ax = plt.subplots(figsize=(15, 4))
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        sns.kdeplot(data=df, x='Age', hue='satisfaction', fill=True, common_norm=False, palette=['#f87171', '#34d399'], alpha=0.4, ax=ax)
        ax.set_title('Age Density curves for Satisfied vs Dissatisfied Passengers', color='#f8fafc')
        ax.tick_params(colors='#94a3b8')
        ax.xaxis.label.set_color('#94a3b8')
        ax.yaxis.label.set_color('#94a3b8')
        st.pyplot(fig)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Data file not found under data/ directory. Place the CSV file to display live statistics.")

# ----------------- TAB 3: MODEL PERFORMANCE -----------------
with tab3:
    st.markdown('<div class="section-header">Machine Learning Model Diagnostics</div>', unsafe_allow_html=True)
    
    col_metric1, col_metric2 = st.columns(2)
    
    with col_metric1:
        st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
        st.write("### 🎯 Classifier Statistics (Optimized Model)")
        
        metrics_data = {
            "Metric Parameter": ["Model Classifier", "Dataset Size", "Training Records", "Testing Records", "Model Accuracy", "Model Precision", "Model Recall", "F1-Score"],
            "Value Metric": ["Random Forest (10 Features)", "15,000 passenger logs", "12,000 logs (80%)", "3,000 logs (20%)", "98.03%", "98.10%", "97.90%", "98.00%"]
        }
        metrics_df = pd.DataFrame(metrics_data)
        st.table(metrics_df)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_metric2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.write("### 📊 Top Feature Importance Scores")
        
        top_features = ["Online_boarding", "Cleanliness", "Seat_comfort", "Inflight_entertainment", 
                        "Inflight_wifi_service", "Ease_of_Online_booking", "Flight_Distance", 
                        "On-board_service", "Class", "Arrival_Delay_in_Minutes"]
        importance_scores = [0.1513, 0.1382, 0.1234, 0.1162, 0.0831, 0.0703, 0.0570, 0.0511, 0.0398, 0.0352]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        sns.barplot(x=importance_scores, y=top_features, palette='viridis', ax=ax)
        ax.set_title('Feature Importance (Optimized Model)', color='#f8fafc')
        ax.tick_params(colors='#94a3b8')
        ax.xaxis.label.set_color('#94a3b8')
        ax.yaxis.label.set_color('#94a3b8')
        st.pyplot(fig)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)
