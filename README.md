# 🩺 Diabetes Prediction using Logistic Regression  
A complete end-to-end **Data Cleaning → EDA → Outlier Handling → Standardization → SMOTE → Model Training** pipeline using the Pima Indians Diabetes Dataset.

---

## 📌 **Project Overview**
This project builds a **binary classification model** to predict whether a person has diabetes based on several medical attributes such as glucose levels, insulin, BMI, age, etc.

The dataset contains many **missing or zero values**, right-skewed distributions, and class imbalance.  
The goal is to clean the data and build a robust model using **Logistic Regression**.

---

## 📂 **Workflow Summary**
### ✔ Step 1: Importing Libraries & Dataset  
- Loaded dataset using `pandas`  
- Displayed dataset summary  
- Checked missing values and data distribution  

### ✔ Step 2: Zero Value Treatment  
Some columns contain 0 values that are **not logically possible** (like BMI = 0).  
We replaced zeros using:
- **Median** for right-skewed features  
- **Mean** for normally distributed features  

| Right-skewed (use Median) | Normal (use Mean) |
|----------------------------|--------------------|
| Pregnancies                | Glucose           |
| Insulin                    | BloodPressure     |
| DiabetesPedigreeFunction   | SkinThickness     |
| Age                        | BMI               |

---

## ✔ Step 3: Outlier Detection & Handling  
Two methods used:

### 🔹 IQR Method (for skewed features)
Used for:
- Pregnancies  
- Insulin  
- BMI  
- DiabetesPedigreeFunction  
- Age

Extreme values capped at lower & upper whiskers.

### 🔹 Z-score Method (for normally distributed features)
Used for:
- Glucose  
- BloodPressure  
- SkinThickness

Values beyond ±3 standard deviations were capped.

---

## ✔ Step 4: Standardization  
Used `StandardScaler()` to bring all features to a similar scale, improving model performance.

Generated a boxplot after scaling to confirm outlier reduction.

---

## ✔ Step 5: Train-Test Split  
Split the cleaned dataset:

```python
test_size = 0.33
random_state = 42
✔ Step 6: Handling Class Imbalance (SMOTE)
The dataset is imbalanced:

More non-diabetic samples than diabetic ones

Used SMOTE to oversample minority class.

✔ Step 7: Logistic Regression Model
Trained logistic regression on resampled training data

Predicted on test set

Evaluated using:

🔹 Accuracy
🔹 Classification Report
🔹 Precision, Recall, F1-score
📊 Model Performance
The final classification report includes metrics for:

Non-diabetic

Diabetic

Helps understand how well the model performs on both classes.

🛠️ Technologies Used
Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

imbalanced-learn (SMOTE)

📁 Project Structure
Copy code
├── diabetes.csv
├── diabetes_prediction.ipynb / .py
├── README.md
🚀 How to Run the Project
Upload the dataset in Google Colab or your environment.

Install required libraries:

bash
Copy code
pip install numpy pandas seaborn scikit-learn imbalanced-learn
Run all cells in the notebook.

⭐ Future Improvements
Try other ML models (Random Forest, XGBoost, SVM)

Compare ROC-AUC curves

Add hyperparameter tuning

Deploy model using Flask/Streamlit

🤝 Contributions
Feel free to fork the repository and submit pull requests.

📬 Contact
Created by Sarthak Shukla

GitHub: sarthakshukla74

LinkedIn: Sarthak Shukla
