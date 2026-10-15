# ✈️ AeroFlow AI: Airline Passenger Satisfaction Predictor & Experience Analytics

An end-to-end Machine Learning project to predict airline passenger satisfaction (Satisfied vs. Neutral/Dissatisfied) based on passenger demographics, flight profiles, operational flight delays, and inflight service ratings.

---

## 🌟 Key Project Highlights

1. **Benchmark Aviation Dataset**: Analyzes a comprehensive 15,000-passenger dataset from Kaggle containing customer demographics, flight parameters, and 14 detailed inflight and ground service ratings.
2. **Exploratory Data Analysis (EDA)**: Interactive Jupyter Notebook featuring 6 critical visualizations:
   - Target variable class distribution (Semi-balanced 57:43 ratio).
   - Satisfaction distribution split across travel classes (Business vs. Economy).
   - Satisfaction split across travel purposes (Business travel vs. Personal travel).
   - Passenger age profile density curve (KDE distribution).
   - Outlier detection on operational flight delays using boxplots.
   - Comprehensive correlation heatmap detailing feature interactions and delay multicollinearity.
3. **Dimensionality Reduction & Feature Selection**:
   - Filtered dataset complexity down from 22 initial features to the **Top 10 most impactful features** using Random Forest Gini Importance.
   - The selected top 10 features capture **86.56% of the total predictive power** while drastically speeding up inference.
4. **Machine Learning Model Comparison**:
   - **Baseline Model:** Logistic Regression (`97.93%` Test Accuracy, `0.98` F1-Score).
   - **Production Model:** Random Forest Classifier (`98.07%` Test Accuracy, `0.98` F1-Score, only 58 errors out of 3,000 test logs).
5. **Interactive Streamlit Web Portal**:
   - Modern dark glassmorphic interface (`style.css`).
   - Dynamic form with unselected placeholder states (`Select Class`, `Select Rating`, `None`) and mandatory input validation.
   - Real-time prediction with confidence probability scores.
   - **Prescriptive Recommendation Engine:** Automatically highlights specific service failure points (< 3 stars) and offers actionable retention advice.
   - Live analytics dashboard displaying key operational KPIs and dynamic Seaborn charts.

---

## 📂 Project Directory Structure

```text
airline_satisfaction_project/
├── app/
│   ├── app.py                            # Streamlit web application
│   └── style.css                         # Custom dark glassmorphic stylesheet
├── data/
│   └── airline_passenger_satisfaction.csv# 15,000 passenger flight records
├── notebooks/
│   └── airline_satisfaction_training.ipynb# Interactive Jupyter Notebook (6 EDA plots + ML Pipeline)
├── models/
│   ├── passenger_satisfaction_model.pkl  # Trained Random Forest classifier (.pkl)
│   ├── scaler.pkl                        # Fitted StandardScaler object (.pkl)
│   └── metadata.pkl                      # Feature order & category mapping metadata
├── requirements.txt                      # Project Python dependencies
├── train_model.py                        # Standalone production retraining script
└── README.md                             # Project documentation (this file)
```

---

## 🛠️ Setup & Running Instructions

### 1. Prerequisites
Ensure you have Python installed on your system (Python 3.8 to 3.12 recommended).

### 2. Clone Repository & Open Workspace
```bash
git clone https://github.com/<your-username>/airline-passenger-satisfaction.git
cd airline-passenger-satisfaction
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Jupyter Training Notebook
To explore the EDA visualizations and training process interactively:
```bash
jupyter notebook notebooks/airline_satisfaction_training.ipynb
```

### 5. Launch the Streamlit Web Application
To start the live interactive web portal:
```bash
streamlit run app/app.py
```
The application will automatically open in your default browser at `http://localhost:8501`.

### 6. (Optional) Retrain the Machine Learning Model
To retrain the Random Forest model and regenerate serialized assets inside `models/`:
```bash
python train_model.py
```

---

## 📊 Model Performance Metrics

| Evaluation Metric | Baseline (Logistic Regression) | Production (Random Forest) |
| :--- | :---: | :---: |
| **Test Accuracy** | 97.93% | **98.07%** |
| **Precision (Macro Avg)** | 0.98 | **0.98** |
| **Recall (Macro Avg)** | 0.98 | **0.98** |
| **F1-Score (Macro Avg)** | 0.98 | **0.98** |
| **Total Test Samples** | 3,000 | 3,000 |
| **Misclassifications** | 62 errors | **58 errors** |

### 🏆 Top Feature Drivers (Gini Importance):
1. **Online Boarding** (15.13%)
2. **Cleanliness** (13.82%)
3. **Seat Comfort** (12.34%)
4. **Inflight Entertainment** (11.62%)
5. **Inflight Wifi Service** (8.31%)
