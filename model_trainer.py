from sklearn.ensemble import RandomForestClassifier
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
        using Wrapper Method (RandomForest feature importance)
        """
        # Step 1: Train a Random Forest Classifier to compute feature importances
        rf = RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced"
        )

        # Step 2: Feature selection based on feature importance from RandomForest
        feature_selector = SelectFromModel(
            rf, threshold="mean", max_features="auto"
        )  # Select features with importance greater than the mean

        # Step 3: Create pipeline with scaling and feature selection
        self.pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),  # Normalize features
                ("feature_selection", feature_selector),  # Feature selection step
                ("classifier", rf),  # Classifier with balanced class weights
            ]
        )

        # Train the pipeline on the training data
        self.pipeline.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the trained model on the test data
        """
        # Predict on the test data
        y_pred = self.pipeline.predict(X_test)

        # Generate classification report and confusion matrix
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        return report, conf_matrix

    def save_model(self, filepath):
        """
        Save the trained model to a pickle file
        """
        import pickle

        with open(filepath, "wb") as f:
            pickle.dump(self.pipeline, f)

    def load_model(self, filepath):
        """
        Load a previously trained model from a pickle file
        """
        import pickle

        with open(filepath, "rb") as f:
            self.pipeline = pickle.load(f)
