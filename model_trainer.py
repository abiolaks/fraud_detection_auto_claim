import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


class ModelTrainer:
    def __init__(self):
        self.pipeline = None

    def train_model(self, X_train, y_train):
        """
        Train a machine learning model using RandomForestClassifier with class_weight='balanced' and feature selection
        """
        # Embedded Feature Selection using Logistic Regression with L1 penalty
        feature_selector = SelectFromModel(
            LogisticRegression(penalty="l1", solver="liblinear")
        )

        # RandomForest with balanced class weights to address class imbalance
        classifier = RandomForestClassifier(class_weight="balanced")

        # Create pipeline
        self.pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("feature_selection", feature_selector),
                ("classifier", classifier),
            ]
        )

        # Train the model
        self.pipeline.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the trained model on the test data
        """
        y_pred = self.pipeline.predict(X_test)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        return report, conf_matrix

    def save_model(self, filepath):
        """
        Save the trained model to a pickle file
        """
        with open(filepath, "wb") as f:
            pickle.dump(self.pipeline, f)

    def load_model(self, filepath):
        """
        Load a previously trained model from a pickle file
        """
        with open(filepath, "rb") as f:
            self.pipeline = pickle.load(f)
