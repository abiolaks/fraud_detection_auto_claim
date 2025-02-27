import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


class Preprocessor:
    def __init__(self):
        self.le = LabelEncoder()
        self.imputer = SimpleImputer(strategy="most_frequent")

    def preprocess_data(self, df):
        """
        Preprocess the input dataframe by:
        - Dropping duplicates
        - Handling missing values
        - Encoding categorical variables using LabelEncoder
        """
        # Drop duplicates
        df = df.drop_duplicates()

        # Handle missing values: impute categorical with mode and numeric with mean
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = self.imputer.fit_transform(df[[col]])
            else:
                df[col] = df[col].fillna(df[col].mean())

        # Encode categorical columns using LabelEncoder
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = self.le.fit_transform(df[col])

        return df
