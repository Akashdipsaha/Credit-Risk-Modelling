import streamlit as st
import numpy as np
import pandas as pd
import time
import os
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from scipy.stats import chi2_contingency, f_oneway
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Set page layout
st.set_page_config(page_title="Credit Risk Prediction", layout="wide")

# Title
st.title('Credit Risk Prediction App')

# Load datasets
@st.cache_data
def load_data():
    internal_data_path = "Internal_Bank_Dataset.xlsx"
    external_data_path = "External_Cibil_Dataset.xlsx"
    unseen_data_path = "Unseen_Dataset.xlsx"

    internal_df = pd.read_excel(internal_data_path)
    external_df = pd.read_excel(external_data_path)
    unseen_df = pd.read_excel(unseen_data_path)

    return internal_df, external_df, unseen_df

internal_df, external_df, unseen_df = load_data()

# Data Preprocessing and Merging
st.subheader("Data Preprocessing")

# Copy datasets for manipulation
df1 = internal_df.copy()
df2 = external_df.copy()

# Clean internal dataset
df1 = df1.loc[df1['Age_Oldest_TL'] != -99999]

# Clean external dataset
columns_to_be_removed = []
for col in df2.columns:
    if df2.loc[df2[col] == -99999].shape[0] > 10000:
        columns_to_be_removed.append(col)
df2 = df2.drop(columns_to_be_removed, axis=1)
for col in df2.columns:
    df2 = df2.loc[df2[col] != -99999]

# Merge the two dataframes on PROSPECTID
df = pd.merge(df1, df2, how='inner', on='PROSPECTID')

# Display a preview of merged data
st.write("Preview of merged data:")
st.dataframe(df.head())

# Feature Selection
st.subheader("Feature Selection")

# Chi-Square test for categorical features
categorical_columns = ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']
selected_cat_features = []
for col in categorical_columns:
    chi2, pval, _, _ = chi2_contingency(pd.crosstab(df[col], df['Approved_Flag']))
    if pval <= 0.05:
        selected_cat_features.append(col)

# Variance Inflation Factor (VIF) for numerical features
numeric_columns = [col for col in df.columns if df[col].dtype != 'object' and col not in ['PROSPECTID', 'Approved_Flag']]
columns_to_be_kept = []

vif_data = df[numeric_columns].copy()
for i in range(len(numeric_columns)):
    vif_value = variance_inflation_factor(vif_data.values, i)
    if vif_value <= 6:
        columns_to_be_kept.append(numeric_columns[i])

# ANOVA test for numerical features
selected_num_features = []
for col in columns_to_be_kept:
    a = list(df[col])
    b = list(df['Approved_Flag'])
    groups = [value for value, group in zip(a, b)]
    f_statistic, p_value = f_oneway(*[value for value, group in zip(a, b) if group in ['P1', 'P2', 'P3', 'P4']])
    if p_value <= 0.05:
        selected_num_features.append(col)

# Final feature list
features = selected_num_features + selected_cat_features
df = df[features + ['Approved_Flag']]

# Encoding categorical variables
df = pd.get_dummies(df, columns=selected_cat_features)

# Train-Test Split
X = df.drop('Approved_Flag', axis=1)
y = df['Approved_Flag']

# Model Training
st.subheader("Model Training and Hyperparameter Tuning")

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define XGBoost model
xgb_classifier = xgb.XGBClassifier(objective='multi:softmax', num_class=4, colsample_bytree=0.9, learning_rate=1, max_depth=3, alpha=10, n_estimators=100)

# Train the model
xgb_classifier.fit(x_train, y_train)

# Predict on test data
y_pred = xgb_classifier.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

# Display accuracy and classification report
st.write(f'Accuracy: {accuracy:.2f}')
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
st.write(f'Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1_score:.2f}')

# Feature to make predictions on unseen data
st.subheader("Prediction on New Data")
df_unseen = unseen_df.copy()

# Preprocess unseen data similarly
df_unseen = pd.get_dummies(df_unseen, columns=selected_cat_features)
df_unseen = df_unseen.reindex(columns=X.columns, fill_value=0)

# Predict on unseen data
y_pred_unseen = xgb_classifier.predict(df_unseen)

# Add predictions to the unseen dataset
unseen_df['Credit_Risk_Prediction'] = y_pred_unseen

# Download the results
st.write("Download the prediction results:")
st.download_button("Download Predictions", unseen_df.to_csv(index=False), file_name="Credit_Risk_Prediction.csv")

# Filter Section for User Interaction
st.subheader("Filter Data by Features")
marital_status_filter = st.selectbox('Select Marital Status', options=unseen_df['MARITALSTATUS'].unique())
education_filter = st.selectbox('Select Education Level', options=unseen_df['EDUCATION'].unique())
gender_filter = st.selectbox('Select Gender', options=unseen_df['GENDER'].unique())

# Filter unseen dataset
filtered_df = unseen_df[
    (unseen_df['MARITALSTATUS'] == marital_status_filter) &
    (unseen_df['EDUCATION'] == education_filter) &
    (unseen_df['GENDER'] == gender_filter)
]

# Display filtered data
st.write("Filtered Data:")
st.dataframe(filtered_df)

# Predict filtered data
filtered_encoded = pd.get_dummies(filtered_df, columns=selected_cat_features)
filtered_encoded = filtered_encoded.reindex(columns=X.columns, fill_value=0)
filtered_predictions = xgb_classifier.predict(filtered_encoded)

# Display prediction for filtered data
filtered_df['Credit_Risk_Prediction'] = filtered_predictions
st.write("Filtered Data with Predictions:")
st.dataframe(filtered_df)

# Save final results
filtered_df.to_csv("Final_Prediction_Filtered.csv", index=False)
st.download_button("Download Filtered Predictions", filtered_df.to_csv(index=False), file_name="Final_Prediction_Filtered.csv")
