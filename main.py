# Streamlit App
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, accuracy_score, classification_report, precision_recall_fscore_support
from scipy.stats import chi2_contingency, f_oneway
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
import warnings
import os
import time

# Streamlit title
st.title("Credit Card Approval Prediction Model")

st.write("This app uses machine learning models to predict credit card approval using internal and external datasets.")

# Load the datasets
st.write("Loading datasets...")

@st.cache_data
def load_data():
    a1 = pd.read_excel("Internal_Bank_Dataset.xlsx")
    a2 = pd.read_excel("External_Cibil_Dataset.xlsx")
    a3 = pd.read_excel("Unseen_Dataset.xlsx")
    return a1, a2, a3

a1, a2, a3 = load_data()

df1 = a1.copy()
df2 = a2.copy()

# Remove Nulls
df1 = df1.loc[df1['Age_Oldest_TL'] != -99999]

columns_to_be_removed = []
for i in df2.columns:
    if df2.loc[df2[i] == -99999].shape[0] > 10000:
        columns_to_be_removed.append(i)
        
df2 = df2.drop(columns_to_be_removed, axis=1)

for i in df2.columns:
    df2 = df2.loc[df2[i] != -99999]

# Merge the two dataframes, inner join so that no nulls are present
df = pd.merge(df1, df2, how='inner', left_on=['PROSPECTID'], right_on=['PROSPECTID'])

# Checking how many are Categorical data 
st.write("Performing feature engineering...")

# Chi-square test
st.write("Chi-Square Test Results:")
for i in ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']:
    chi2, pval, _, _ = chi2_contingency(pd.crosstab(df[i], df['Approved_Flag']))
    st.write(f"{i}: {pval}")

# VIF for numerical columns
numeric_columns = [i for i in df.columns if df[i].dtype != 'object' and i not in ['PROSPECTID','Approved_Flag']]
vif_data = df[numeric_columns]
columns_to_be_kept = []

for i in range(vif_data.shape[1]):
    vif_value = variance_inflation_factor(vif_data, i)
    if vif_value <= 6:
        columns_to_be_kept.append(numeric_columns[i])
    else:
        vif_data = vif_data.drop([numeric_columns[i]], axis=1)

# Anova test for numeric columns
st.write("ANOVA Test Results:")
columns_to_be_kept_numerical = []

for i in columns_to_be_kept:
    a = list(df[i])  
    b = list(df['Approved_Flag'])  
    
    group_P1 = [value for value, group in zip(a, b) if group == 'P1']
    group_P2 = [value for value, group in zip(a, b) if group == 'P2']
    group_P3 = [value for value, group in zip(a, b) if group == 'P3']
    group_P4 = [value for value, group in zip(a, b) if group == 'P4']

    f_statistic, p_value = f_oneway(group_P1, group_P2, group_P3, group_P4)
    
    if p_value <= 0.05:
        columns_to_be_kept_numerical.append(i)

# Final features
features = columns_to_be_kept_numerical + ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']
df = df[features + ['Approved_Flag']]

# Label encoding for categorical features
df['MARITALSTATUS'].replace({'SSC': 1, '12TH': 2, 'GRADUATE': 3, 'UNDER GRADUATE': 3, 'POST-GRADUATE': 4, 'OTHERS': 1, 'PROFESSIONAL': 3}, inplace=True)

# Encoding
df_encoded = pd.get_dummies(df, columns=['MARITALSTATUS','GENDER', 'last_prod_enq2' ,'first_prod_enq2'])

# Model training - Random Forest
st.write("Training Random Forest Classifier...")

y = df_encoded['Approved_Flag']
x = df_encoded.drop(['Approved_Flag'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=200, random_state=42)
rf_classifier.fit(x_train, y_train)

y_pred = rf_classifier.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
st.write(f"Random Forest Accuracy: {accuracy:.2f}")

# XGBoost
st.write("Training XGBoost Classifier...")

model = xgb.XGBClassifier(objective='multi:softmax', num_class=4, colsample_bytree=0.9, learning_rate=1, max_depth=3, alpha=10, n_estimators=100)
model.fit(x_train, y_train)
y_pred_unseen = model.predict(df_encoded)

# Display Final Prediction
st.write("Final Prediction on Unseen Dataset:")
df_unseen = a3.copy()
df_unseen['Target_variable'] = y_pred_unseen
st.dataframe(df_unseen)

# Save the prediction to file
df_unseen.to_excel("Final_Prediction.xlsx", index=False)
st.write("Predictions saved to 'Final_Prediction.xlsx'!")
