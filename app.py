import streamlit as st
import joblib
import pandas as pd

# Load pipelines
regressor_pipeline = joblib.load('regressor_pipeline.pkl')
classifier_pipeline_Pass_Fail = joblib.load('classifier_pipeline_Pass_Fail Status.pkl')
classifier_pipeline_Engagement = joblib.load('classifier_pipeline_Engagement Level.pkl')
classifier_pipeline_Dropout = joblib.load('classifier_pipeline_Dropout Likelihood.pkl')

# Get expected features for each pipeline
regressor_features = regressor_pipeline.named_steps['preprocessor'].get_feature_names_out()
pass_fail_features = classifier_pipeline_Pass_Fail.named_steps['preprocessor'].get_feature_names_out()
engagement_features = classifier_pipeline_Engagement.named_steps['preprocessor'].get_feature_names_out()
dropout_features = classifier_pipeline_Dropout.named_steps['preprocessor'].get_feature_names_out()

# Print expected features for debugging (optional)
print("Regressor Features:", regressor_features)
print("Pass/Fail Features:", pass_fail_features)
print("Engagement Features:", engagement_features)
print("Dropout Features:", dropout_features)

# User input

st.title("Student Performance Prediction")
st.write("Enter student data:")

# Gather input fields for numerical features
input_data = {
    "Age": st.number_input("Age", min_value=10, max_value=100, value=18),
    "Attendance Percentage": st.number_input("Attendance Percentage", min_value=0.0, max_value=100.0, value=75.0),
    "Previous Academic Records (GPA)": st.number_input("GPA", min_value=0.0, max_value=4.0, value=3.0),
    "Class Assignments Score": st.number_input("Assignment Score", min_value=0.0, max_value=100.0, value=85.0),
    "Login Frequency": st.number_input("Login Frequency", min_value=0, max_value=100, value=10),
    "Time Spent Studying Outside Class (mins)": st.number_input("Study Time (minutes)", min_value=0, max_value=1440, value=120),
    "Proximity to Institute (mins)": st.number_input("Proximity to Institute (minutes)", min_value=0, max_value=120, value=30),}

# Input fields for categorical features
input_data.update({
    "Gender": st.selectbox("Gender", ["Male", "Female"]),
    "Socioeconomic Status": st.selectbox("Socioeconomic Status", ["Low", "Middle", "High"]),
    "Participation in Class Activities": st.selectbox("Participation in Class Activities", ["Active", "Inactive"]),
    "Submissions": st.selectbox("Submissions", ["Early", "On-time", "Late", "No Submissions"]),
    "Motivational Survey Scores": st.selectbox("Motivational Survey Scores", ["Low", "Moderate", "High"]),
    "Stress Levels": st.selectbox("Stress Levels", ["Low", "Moderate", "High"]),
    "Access to Resources": st.selectbox("Access to Resources", ["Adequate", "Inadequate"]),
    "Part-Time Job Status": st.selectbox("Part-Time Job Status", ["Yes", "No"]),
    "Classroom Environment Satisfaction": st.selectbox("Classroom Environment Satisfaction", ["Satisfied", "Neutral", "Unsatisfied"]),
    "Group Learning Sessions": st.selectbox("Group Learning Sessions", ["Frequent", "Rare", "Never"]),
})


input_df = pd.DataFrame([input_data])

# Function to align input_df with pipeline features
def align_features(input_df, expected_features):
    for feature in expected_features:
        if feature not in input_df.columns:
            input_df[feature] = 0  # Default value for missing features
    return input_df[expected_features]  # Reorder columns to match expected order

# Align input_df for each pipeline
input_df_regressor = align_features(input_df, regressor_features)
input_df_pass_fail = align_features(input_df, pass_fail_features)
input_df_engagement = align_features(input_df, engagement_features)
input_df_dropout = align_features(input_df, dropout_features)

if st.button("Predict"):
    # Predict GPA
    gpa_prediction = regressor_pipeline.predict(input_df)[0]
    st.write(f"Predicted Final GPA: {gpa_prediction}")

    # Predict Pass/Fail
    pass_fail_prediction = classifier_pipeline_Pass_Fail.predict(input_df)[0]
    st.write(f"Predicted Pass/Fail Status: {pass_fail_prediction}")

    # Predict Engagement Level
    engagement_prediction = classifier_pipeline_Engagement.predict(input_df)[0]
    st.write(f"Predicted Engagement Level: {engagement_prediction}")

    # Predict Dropout Likelihood
    dropout_prediction = classifier_pipeline_Dropout.predict(input_df)[0]
    st.write(f"Predicted Dropout Likelihood: {dropout_prediction}")


