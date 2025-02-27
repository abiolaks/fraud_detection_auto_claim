from app import FraudDetectionApp
import pandas as pd


def main():
    # Initialize the application
    app = FraudDetectionApp()

    # Train the model (assuming you have a preprocessed dataset to train with)
    # Here we use 'train_data.csv' as a placeholder for the actual dataset file path
    train_data = pd.read_csv("train_data.csv")
    X_train = app.preprocessor.preprocess_data(
        train_data.drop("fraud_reported", axis=1)
    )
    y_train = train_data["fraud_reported"]

    app.trainer.train_model(X_train, y_train)

    # Evaluate the model (you may use a test dataset here)
    test_data = pd.read_csv("test_data.csv")
    X_test = app.preprocessor.preprocess_data(test_data.drop("fraud_reported", axis=1))
    y_test = test_data["fraud_reported"]
    report, conf_matrix = app.trainer.evaluate_model(X_test, y_test)

    print(report)
    print(conf_matrix)

    # Save the trained model
    app.trainer.save_model("fraud_detection_model.pkl")

    # Run the Streamlit app
    app.run()


if __name__ == "__main__":
    main()
