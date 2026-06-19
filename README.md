# ML Prediction Models

A comprehensive collection of machine learning models for predicting various real-world outcomes in healthcare and real estate.

---

## 🩺 Project 1: Diabetes Prediction

**Objective:** To predict whether a patient is diabetic or not based on diagnostic medical measurements such as Glucose levels, BMI, Age, and Insulin.

### 🛠️ Steps Implemented:

1. **Data Loading & Preparation 🧹**
   - Loaded the dataset (`diabetes.csv`).
   - Dataset contains 768 records with 9 columns (8 features + 1 target).
   - Separated the independent medical features from the target variable (`Outcome`).
   - Features include: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, and Age.

2. **Feature Scaling ⚖️**
   - Applied `StandardScaler` to normalize all features to have mean=0 and standard deviation=1.
   - This is crucial because features like 'Glucose' (0-199) and 'DiabetesPedigreeFunction' (0-2.5) have very different ranges, and scaling ensures the model doesn't give more weight to features with larger numeric values.

3. **Data Splitting & Stratification 📊**
   - Split the data into Training and Testing sets using 80-20 ratio (`test_size=0.2`).
   - Used `stratify=y` to maintain the same proportion of diabetic and non-diabetic patients in both training and test sets.
   - Applied `random_state=2` for reproducibility.

4. **Model Selection & Training 🤖**
   - **Algorithm Used:** `Support Vector Machine (SVM)` with Linear Kernel
   - **Configuration:** `kernel='linear'`
   - **Why?** For a binary classification problem like diabetes prediction, SVM with a linear kernel creates a clear decision boundary to separate diabetic from non-diabetic patients. Linear kernel works well when the data is already in a reasonably high-dimensional space.

5. **Evaluation & Results 📊**
   - **Training Accuracy:** ~77.20%
   - **Test Accuracy:** ~70.79%
   - **Conclusion:** The model shows good generalization with minimal overfitting. The test accuracy of 70.79% is a realistic benchmark for real-world medical predictions, indicating the model learns actual patterns rather than memorizing training data.

**How to Use This Model:**
```python
# Load your patient data
patient_data = [[6, 148, 72, 35, 0, 33.6, 0.627, 50]]  
# [Pregnancies, Glucose, BP, Skin, Insulin, BMI, DPF, Age]

# Scale the data using the trained scaler
patient_scaled = scaler.transform(patient_data)

# Make prediction
prediction = model.predict(patient_scaled)
print(f"Diabetes Prediction: {prediction}")  # Output: 0 (No Diabetes) or 1 (Diabetes)
```

---

## 🍷 Project 2: Wine Quality Prediction

**Objective:** To predict the quality of wine (rated on a scale from 3 to 8) based on its chemical properties such as acidity, pH, alcohol, and sugar levels.

### 🛠️ Steps Implemented:

1. **Data Loading & Cleaning 🧹**
   - Loaded the wine quality dataset (`winequality.csv`).
   - Total records: 1,607 wine samples with 11 chemical features.
   - Handled missing data by imputing null values in columns with their respective column medians.
   - Features include: Fixed Acidity, Volatile Acidity, Citric Acid, Residual Sugar, Chlorides, Free Sulfur Dioxide, Total Sulfur Dioxide, Density, pH, Sulphates, and Alcohol.

2. **Data Splitting & Scaling ⚖️**
   - Split the data into 90% training and 10% testing (`test_size=0.1`).
   - **Training Data:** 1,446 samples
   - **Testing Data:** 161 samples
   - Used `stratify=y` to maintain the same quality distribution in both sets.
   - Applied `StandardScaler` to normalize all chemical properties to a common scale.

3. **Model Selection & Training 🤖**
   - **Algorithm Used:** `Random Forest Classifier`
   - **Configuration:** 
     - `n_estimators=100` (100 decision trees)
     - `max_depth=5` (maximum depth of each tree)
     - `min_samples_split=10` (minimum samples to split a node)
     - `min_samples_leaf=5` (minimum samples in leaf node)
     - `max_features='sqrt'` (use square root of features for splitting)
     - `random_state=42` (for reproducibility)
   - **Why?** Wine quality is a **Multi-Class Classification** problem. Random Forest works exceptionally well here because it can capture complex interactions between chemical properties and quality ratings through multiple decision trees voting together.

4. **Evaluation & Results 📊**
   - **Training Accuracy:** 65.14%
   - **Testing Accuracy:** 52.79%
   - **Accuracy Gap:** 12.35%
   - **Conclusion:** The test accuracy of 52.79% is realistic for this subjective multi-class problem. Wine quality ratings are influenced by many overlapping chemical properties, and achieving perfect accuracy would indicate overfitting to training data rather than true generalization.

**How to Use This Model:**
```python
# Wine chemical properties
wine_data = [[7.4, 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]]
# [Fixed Acidity, Volatile Acidity, Citric Acid, Residual Sugar, Chlorides, 
#  Free Sulfur Dioxide, Total Sulfur Dioxide, Density, pH, Sulphates, Alcohol]

# Scale the data
wine_scaled = scaler.transform(wine_data)

# Make prediction
quality = model.predict(wine_scaled)
print(f"Wine Quality Rating: {quality}")  # Output: Quality score (3-8)
```

---

## 🫀 Project 3: Heart Disease Prediction

**Objective:** To predict the presence of heart disease in a patient based on their medical parameters using Machine Learning.

### 🛠️ Steps Implemented:

1. **Data Loading & Cleaning 🧹**
   - Loaded the heart disease dataset.
   - Handled missing values using `SimpleImputer` with strategy='mean' or median replacement.
   - Cleaned categorical columns by encoding string values into numeric form using `LabelEncoder`.
   - Dataset contains 13 medical features including age, sex, chest pain type, blood pressure, and other vital signs.

2. **Feature Encoding & Scaling ⚖️**
   - Applied `LabelEncoder` to convert categorical features (like sex: male/female) into numeric values (0/1).
   - Standardized all features using `StandardScaler` to ensure features with larger numeric ranges don't dominate the model.

3. **Data Splitting & Stratification 📊**
   - Split the data into 80% training and 20% testing (`test_size=0.2`).
   - Used `stratify=y` to maintain the same disease prevalence in both sets.
   - Applied `random_state=42` for reproducible results.

4. **Model Selection & Training 🤖**
   - **Algorithm Used:** `Random Forest Classifier`
   - **Configuration:**
     - `n_estimators=100` (100 decision trees)
     - `max_depth=5` (controls tree complexity)
     - `min_samples_split=10` (prevents overfitting)
     - `min_samples_leaf=5` (ensures generalization)
     - `random_state=42` (reproducibility)
   - **Why?** Random Forest effectively captures the non-linear relationships between medical parameters and heart disease presence. Multiple trees voting together create robust predictions without overfitting.

5. **Evaluation & Results 📊**
   - **Training Accuracy:** 82.50%
   - **Testing Accuracy:** 68.00%
   - **Accuracy Gap:** 14.50%
   - **Conclusion:** The test accuracy of 68% is a reliable and practical benchmark. The 14.5% gap between training and test accuracy shows the model avoids overfitting while maintaining good generalization capability for unseen patient data.

**How to Use This Model:**
```python
# Patient health data (13 features)
patient_data = [[52, 1, 2, 130, 240, 0, 1, 150, 0, 1.2, 1, 0, 2]]  
# [Age, Sex, Chest Pain Type, Resting BP, Cholesterol, Fasting BS, Resting ECG,
#  Max Heart Rate, Exercise Induced Angina, ST Depression, ST Slope, Major Vessels, Thalassemia]

# Scale the data
patient_scaled = scaler.transform(patient_data)

# Make prediction
risk = model.predict(patient_scaled)
print(f"Heart Disease Risk: {risk}")  # Output: 0 (No Disease) or 1 (Disease Present)
```

---

## 🏠 Project 4: House Price Prediction (Regression)

**Objective:** To predict the exact market price of a house based on property attributes like square footage, number of rooms, house age, and location quality.

### 🛠️ Steps Implemented:

1. **Data Preparation 🧹**
   - Loaded the real estate dataset containing 997 property records.
   - Isolated the continuous target variable (`price`) from predictive features.
   - Features include: Square Footage, Bedrooms, Bathrooms, Age, Price per Square Foot, Lot Size, and Neighborhood Quality.
   - Checked for missing values and ensured data quality before model training.

2. **Feature Scaling ⚖️**
   - Applied `StandardScaler` to normalize all property features.
   - This prevents features with large numeric ranges (like square footage: 1000-5000) from overpowering smaller features (like age: 1-100).

3. **Data Splitting 📊**
   - Split the data into 80% training and 20% testing (`test_size=0.2`).
   - **Training Data:** 797 samples
   - **Testing Data:** 200 samples
   - Applied `random_state=2` for reproducible results.

4. **Model Selection & Training 🤖**
   - **Algorithm Used:** `Random Forest Regressor`
   - **Configuration:**
     - `n_estimators=200` (ensemble of 200 decision trees)
     - `max_depth=10` (allows deeper trees to capture complex patterns)
     - `random_state=2` (reproducibility)
   - **Why?** House prices are determined by complex interactions between multiple features. Random Forest with 200 trees captures non-linear relationships (e.g., how location quality multiplies the value of square footage) that linear models cannot capture.

5. **Evaluation & Results 📊**
   - **Training R² Score:** 0.9914 (99.14%)
   - **Testing R² Score:** 0.9438 (94.38%)
   - **Conclusion:** Outstanding performance! R² of 0.9438 means the model explains 94.38% of the variance in house prices on unseen test data. This demonstrates the model is production-ready and highly reliable for real estate price predictions.

**How to Use This Model:**
```python
# House property features (7 features)
house_data = [[2000, 4, 2, 10, 150, 5000, 8]]
# [Square Footage, Bedrooms, Bathrooms, Age, Price/Sqft, Lot Size, Neighborhood Quality]

# Scale the data
house_scaled = scaler.transform(house_data)

# Make prediction
price = model.predict(house_scaled)
print(f"Predicted House Price: ${price[0]:,.2f}")  # Output: Estimated price in dollars
```

---

