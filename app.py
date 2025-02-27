import streamlit as st
import pandas as pd
from model_trainer import ModelTrainer
from preprocessor import Preprocessor


class FraudDetectionApp:
    def __init__(self):
        self.trainer = ModelTrainer()
        self.preprocessor = Preprocessor()

    def load_and_preprocess_data(self, file):
        """
        Load the file and preprocess it
        """
        data = pd.read_csv(file)
        return self.preprocessor.preprocess_data(data)

    def batch_prediction(self, uploaded_file):
        """
        Perform batch prediction from the uploaded CSV file
        """
        test_data = pd.read_csv(uploaded_file)
        test_data_preprocessed = self.preprocessor.preprocess_data(test_data)
        predictions = self.trainer.pipeline.predict(test_data_preprocessed)
        test_data["Prediction"] = predictions
        return test_data

    def single_prediction(self, input_data):
        """
        Perform prediction for a single customer
        """
        input_data_preprocessed = self.preprocessor.preprocess_data(
            pd.DataFrame([input_data])
        )
        prediction = self.trainer.pipeline.predict(input_data_preprocessed)
        return prediction

    def run(self):
        """
        Run the Streamlit app for the Fraud Detection System
        """
        st.title("Auto Insurance Fraudulent Claims Detection")
        st.write("This demo provides a UI to make predictions for fraudulent claims.")

        # Batch Prediction - File Upload
        st.sidebar.header("Batch Prediction")
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV file for batch prediction", type=["csv"]
        )
        if uploaded_file is not None:
            predictions_df = self.batch_prediction(uploaded_file)
            st.write("### Batch Prediction Results")
            st.dataframe(predictions_df)

        # Single Customer Prediction - Form
        st.header("Single Customer Prediction")
        st.write("Enter the feature values for a single customer below:")

        # Load CSV for single prediction
        if uploaded_file is not None:
            uploaded_data = pd.read_csv(uploaded_file)
            # Display slider to select customer
            selected_customer_index = st.slider(
                "Select a customer from the uploaded CSV",
                min_value=0,
                max_value=len(uploaded_data) - 1,
                step=1,
            )
            selected_customer = uploaded_data.iloc[selected_customer_index]

            # Display customer data in form
            single_data = {}
            for col in uploaded_data.columns:
                if col != "fraud_reported":  # Exclude target column
                    val = selected_customer[col]
                    single_data[col] = val
                    st.text(f"{col}: {val}")

            if st.button("Predict for Selected Customer"):
                prediction = self.single_prediction(single_data)
                if prediction[0] == 1:
                    st.error("Fraudulent Claim Detected!")
                else:
                    st.success("Claim is Non-Fraudulent.")
