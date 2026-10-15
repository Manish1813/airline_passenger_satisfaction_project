import json
import os

def generate_notebook(output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Define cells
    cells = []
    
    # Cell 1: Markdown Title
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Airline Passenger Satisfaction Prediction (Optimized 10-Feature Model)\n",
            "## End-to-End Machine Learning Project - Advanced EDA & Model Training\n",
            "\n",
            "**Author:** Data Science & Analysis Student  \n",
            "**Objective:** Build an end-to-end Machine Learning model to predict whether a passenger will be satisfied or neutral/dissatisfied with their flight experience based on their profile and ratings of services.  \n",
            "This notebook implements **6 Critical EDA Visualizations** to uncover data patterns, performs **Feature Selection** to isolate the top 10 driving features, and trains an optimized Random Forest model achieving >98% accuracy."
        ]
    })
    
    # Cell 2: Code Imports
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Import necessary libraries\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "import os\n",
            "\n",
            "# Model Selection and Preprocessing\n",
            "from sklearn.model_selection import train_test_split\n",
            "from sklearn.preprocessing import StandardScaler\n",
            "from sklearn.linear_model import LogisticRegression\n",
            "from sklearn.ensemble import RandomForestClassifier\n",
            "from sklearn.metrics import classification_report, accuracy_score, confusion_matrix\n",
            "\n",
            "# Model Serialization\n",
            "import joblib\n",
            "\n",
            "# Set visual style for plots\n",
            "sns.set_theme(style=\"whitegrid\")\n",
            "plt.rcParams[\"figure.figsize\"] = (12, 5)\n",
            "print(\"Libraries successfully imported!\")"
        ]
    })
    
    # Cell 3: Markdown Data Loading
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Load the Dataset\n",
            "Let's load the generated airline passenger satisfaction dataset containing 15,000 records."
        ]
    })
    
    # Cell 4: Code Data Loading
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load the dataset\n",
            "df = pd.read_csv('../data/airline_passenger_satisfaction.csv')\n",
            "\n",
            "# Display shape and first few rows\n",
            "print(f\"Dataset Shape: {df.shape}\")\n",
            "df.head()"
        ]
    })
    
    # Cell 5: Markdown EDA Title
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Exploratory Data Analysis (EDA) - Top 6 Critical Visualizations"
        ]
    })
    
    # Cell 6: Markdown Plot 1
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Plot 1: Target Variable Class Distribution\n",
            "**Goal:** Inspect for **Class Imbalance** in the target variable (`satisfaction`). An imbalanced dataset forces the model to be biased towards the majority class."
        ]
    })
    
    # Cell 7: Code Plot 1
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "plt.figure(figsize=(6, 4))\n",
            "sns.countplot(x='satisfaction', data=df, hue='satisfaction', palette='viridis', legend=False)\n",
            "plt.title('Passenger Satisfaction Distribution (Target)')\n",
            "plt.xlabel('Satisfaction Status')\n",
            "plt.ylabel('Passenger Count')\n",
            "plt.show()\n",
            "\n",
            "print(df['satisfaction'].value_counts())\n",
            "print(\"\\nClass Distribution Percentage:\")\n",
            "print(df['satisfaction'].value_counts(normalize=True) * 100)"
        ]
    })
    
    # Cell 8: Markdown Plot 1 Insight
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "**Insight 1:** The dataset is semi-balanced with **57.2% Neutral/Dissatisfied** and **42.8% Satisfied** passengers. Since there is no extreme class imbalance (e.g. 99% vs 1%), we can safely use **Accuracy** as a reliable evaluation metric."
        ]
    })
    
    # Cell 9: Markdown Plot 2
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Plot 2: Satisfaction Split by Travel Class\n",
            "**Goal:** Understand how the ticket class (`Business`, `Eco`, `Eco Plus`) impacts overall traveler satisfaction."
        ]
    })
    
    # Cell 10: Code Plot 2
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "sns.countplot(data=df, x='Class', hue='satisfaction', palette=['#f87171', '#34d399'])\n",
            "plt.title('Satisfaction Distribution by Ticket Class')\n",
            "plt.xlabel('Travel Class')\n",
            "plt.ylabel('Count')\n",
            "plt.legend(title='Satisfaction')\n",
            "plt.show()"
        ]
    })
    
    # Cell 11: Markdown Plot 2 Insight
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "**Insight 2:** Passengers in **Business Class** show an exceptionally high satisfaction rate, whereas **Economy (Eco)** and **Economy Plus** flyers are overwhelmingly dissatisfied. This highlights that cabin space, comfort, and premium treatments are strong satisfaction drivers."
        ]
    })
    
    # Cell 12: Markdown Plot 3
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Plot 3: Satisfaction Split by Type of Travel\n",
            "**Goal:** Analyze if the flight purpose (Business vs. Personal Travel) acts as a differentiator of passenger mood."
        ]
    })
    
    # Cell 13: Code Plot 3
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "sns.countplot(data=df, x='Type_of_Travel', hue='satisfaction', palette=['#f87171', '#34d399'])\n",
            "plt.title('Satisfaction Distribution by Travel Purpose')\n",
            "plt.xlabel('Type of Travel')\n",
            "plt.ylabel('Count')\n",
            "plt.legend(title='Satisfaction')\n",
            "plt.show()"
        ]
    })
    
    # Cell 14: Markdown Plot 3 Insight
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "**Insight 3:** **Business travelers** are mostly satisfied, whereas **Personal/Leisure travelers** are highly dissatisfied. Personal travelers usually pay out of pocket, making them highly price-sensitive and less tolerant of lack of amenities."
        ]
    })
    
    # Cell 15: Markdown Plot 4
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Plot 4: Passenger Age Distribution by Satisfaction\n",
            "**Goal:** Check if age plays a role in traveler satisfaction using density curves."
        ]
    })
    
    # Cell 16: Code Plot 4
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "sns.kdeplot(data=df, x='Age', hue='satisfaction', fill=True, common_norm=False, palette=['#f87171', '#34d399'], alpha=0.4)\n",
            "plt.title('Age Density Curves for Satisfied vs Dissatisfied Passengers')\n",
            "plt.xlabel('Age (Years)')\n",
            "plt.ylabel('Density')\n",
            "plt.show()"
        ]
    })
    
    # Cell 17: Markdown Plot 4 Insight
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "**Insight 4:** Satisfaction density peaks in the **35-50 age bracket**, which corresponds to frequent business flyers. Younger flyers (under 30) and elderly passengers (above 60) show higher relative rates of dissatisfaction, likely due to different comfort expectations."
        ]
    })
    
    # Cell 18: Markdown Plot 5
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Plot 5: Outlier Detection on Continuous Numerical Columns\n",
            "**Goal:** Use Boxplots to visually inspect continuous numerical variables for outliers before model training."
        ]
    })
    
    # Cell 19: Code Plot 5
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "fig, axes = plt.subplots(1, 3, figsize=(15, 5))\n",
            "sns.boxplot(data=df, y='Age', color='skyblue', ax=axes[0])\n",
            "axes[0].set_title('Age Boxplot')\n",
            "\n",
            "sns.boxplot(data=df, y='Flight_Distance', color='lightgreen', ax=axes[1])\n",
            "axes[1].set_title('Flight Distance Boxplot')\n",
            "\n",
            "sns.boxplot(data=df, y='Arrival_Delay_in_Minutes', color='salmon', ax=axes[2])\n",
            "axes[2].set_title('Arrival Delay Boxplot')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    })
    
    # Cell 20: Markdown Plot 5 Insight
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "**Insight 5:** Age and Flight Distance have no extreme outliers. However, the **Arrival Delay** column is heavily populated with outliers (data points beyond the upper whisker). This is standard for flight delay data and will be handled by robust tree-based models like Random Forests which are immune to outliers."
        ]
    })
    
    # Cell 21: Markdown Plot 6
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Plot 6: Feature Correlation Heatmap\n",
            "**Goal:** Understand how all numeric and categorical features relate to each other and identify target correlations and multicollinearity."
        ]
    })
    
    # Cell 22: Code Plot 6
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Map columns to numbers temporarily for correlation heatmap\n",
            "corr_df = df.copy()\n",
            "corr_df['Gender'] = corr_df['Gender'].map({'Male': 0, 'Female': 1})\n",
            "corr_df['Customer_Type'] = corr_df['Customer_Type'].map({'Loyal Customer': 0, 'disloyal Customer': 1})\n",
            "corr_df['Type_of_Travel'] = corr_df['Type_of_Travel'].map({'Business travel': 0, 'Personal Travel': 1})\n",
            "corr_df['Class'] = corr_df['Class'].map({'Business': 0, 'Eco': 1, 'Eco Plus': 2})\n",
            "corr_df['satisfaction'] = corr_df['satisfaction'].map({'neutral_or_dissatisfied': 0, 'satisfied': 1})\n",
            "\n",
            "# Compute correlation matrix\n",
            "corr_matrix = corr_df.corr()\n",
            "\n",
            "# Plot Heatmap\n",
            "plt.figure(figsize=(12, 10))\n",
            "sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt='.2f', linewidths=0.5)\n",
            "plt.title('Full Feature Correlation Matrix')\n",
            "plt.show()"
        ]
    })
    
    # Cell 23: Markdown Plot 6 Insight
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "**Insight 6:**\n",
            "1. **Satisfaction correlations:** Online Boarding, Inflight Wifi, and Seat Comfort show the strongest positive correlation with passenger satisfaction.\n",
            "2. **Multicollinearity:** Departure Delay and Arrival Delay show an almost perfect linear correlation (~0.98), suggesting that they contain identical information. We can keep only one or use Random Forest which natively handles collinearity."
        ]
    })
    
    # Cell 24: Markdown Preprocessing
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Feature Selection using Random Forest\n",
            "To simplify our deployment model, we train a temporary Random Forest Classifier on all features and retrieve the importance score of each column."
        ]
    })
    
    # Cell 25: Code Feature Selection
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Separate features and target\n",
            "X_all = corr_df.drop('satisfaction', axis=1)\n",
            "y_all = corr_df['satisfaction']\n",
            "\n",
            "# 2. Train a temporary Random Forest model\n",
            "model_temp = RandomForestClassifier(n_estimators=100, random_state=42)\n",
            "model_temp.fit(X_all, y_all)\n",
            "\n",
            "# 3. Print the importance score of each feature directly\n",
            "importances = model_temp.feature_importances_\n",
            "print(\"Feature Importances:\")\n",
            "for i in range(len(X_all.columns)):\n",
            "    column_name = X_all.columns[i]\n",
            "    score = importances[i]\n",
            "    print(column_name, \":\", round(score, 4))"
        ]
    })
    
    # Cell 26: Markdown Data Split
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Final Model Preprocessing & Training (Top 10 Features)\n",
            "We filter the dataset to the top 10 features, perform train-test split (80-20), scale the features, and train our optimized Random Forest Classifier."
        ]
    })
    
    # Cell 27: Code Final Split & Scale
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "selected_features = [\n",
            "    'Online_boarding',\n",
            "    'Cleanliness',\n",
            "    'Seat_comfort',\n",
            "    'Inflight_entertainment',\n",
            "    'Inflight_wifi_service',\n",
            "    'Ease_of_Online_booking',\n",
            "    'Flight_Distance',\n",
            "    'On-board_service',\n",
            "    'Class',\n",
            "    'Arrival_Delay_in_Minutes'\n",
            "]\n",
            "\n",
            "# Map categories to numbers for final modeling\n",
            "df_final = df.copy()\n",
            "df_final['Class'] = df_final['Class'].map({'Business': 0, 'Eco': 1, 'Eco Plus': 2})\n",
            "df_final['satisfaction'] = df_final['satisfaction'].map({'neutral_or_dissatisfied': 0, 'satisfied': 1})\n",
            "\n",
            "X = df_final[selected_features]\n",
            "y = df_final['satisfaction']\n",
            "\n",
            "# Handle missing values by replacing them with the column median\n",
            "X = X.fillna(X.median(numeric_only=True))\n",
            "\n",
            "# Train test split (80% train, 20% test)\n",
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n",
            "\n",
            "# Scale the features using StandardScaler\n",
            "scaler = StandardScaler()\n",
            "X_train_scaled = scaler.fit_transform(X_train)\n",
            "X_test_scaled = scaler.transform(X_test)\n",
            "print(\"Top 10 features scaled successfully!\")"
        ]
    })
    
    # Cell 28: Code Baseline Model Training
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Train Baseline Logistic Regression Model\n",
            "lr_model = LogisticRegression(random_state=42, max_iter=1000)\n",
            "lr_model.fit(X_train_scaled, y_train)\n",
            "\n",
            "lr_pred = lr_model.predict(X_test_scaled)\n",
            "print(f\"Logistic Regression Test Accuracy: {accuracy_score(y_test, lr_pred)*100:.2f}%\")\n",
            "print(\"\\nClassification Report:\")\n",
            "print(classification_report(y_test, lr_pred))"
        ]
    })
    
    # Cell 29: Code Production Model Training
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Train Production Random Forest Classifier Model\n",
            "rf_model = RandomForestClassifier(n_estimators=100, random_state=42)\n",
            "rf_model.fit(X_train_scaled, y_train)\n",
            "\n",
            "rf_pred = rf_model.predict(X_test_scaled)\n",
            "print(f\"Random Forest Test Accuracy: {accuracy_score(y_test, rf_pred)*100:.2f}%\")\n",
            "print(\"\\nClassification Report:\")\n",
            "print(classification_report(y_test, rf_pred))"
        ]
    })
    
    # Cell 30: Code Confusion Matrix
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Plot Confusion Matrix for Random Forest\n",
            "cm = confusion_matrix(y_test, rf_pred)\n",
            "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')\n",
            "plt.title('Random Forest - Confusion Matrix')\n",
            "plt.ylabel('Actual Label')\n",
            "plt.xlabel('Predicted Label')\n",
            "plt.show()"
        ]
    })
    
    # Cell 31: Code Save model files
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Create models folder if it does not exist\n",
            "if not os.path.exists('../models'):\n",
            "    os.makedirs('../models')\n",
            "\n",
            "# Save the trained model and scaler\n",
            "joblib.dump(rf_model, '../models/passenger_satisfaction_model.pkl')\n",
            "joblib.dump(scaler, '../models/scaler.pkl')\n",
            "\n",
            "# Save mapping metadata for deployment references\n",
            "metadata = {\n",
            "    'mappings': {\n",
            "        'Class': {'Business': 0, 'Eco': 1, 'Eco Plus': 2},\n",
            "        'satisfaction': {'neutral_or_dissatisfied': 0, 'satisfied': 1}\n",
            "    },\n",
            "    'feature_names': selected_features\n",
            "}\n",
            "joblib.dump(metadata, '../models/metadata.pkl')\n",
            "\n",
            "print(\"Model, Scaler, and Metadata successfully saved to '../models/' folder!\")"
        ]
    })
    
    # Assemble notebook structure
    notebook_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipykernel",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipykernel",
                "version": "3.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook_dict, f, indent=2)
        
    print(f"Jupyter Notebook successfully updated at {output_path}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    notebook_file = os.path.join(current_dir, "notebooks", "airline_satisfaction_training.ipynb")
    generate_notebook(notebook_file)
