import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


class Preprocessor:
    def __init__(self):
        self.imputer = SimpleImputer(strategy="most_frequent")
        self.ohe = OneHotEncoder(
            sparse=False, drop="first"
        )  # Using drop='first' to avoid multicollinearity

    def preprocess_data(self, df):
        """
        Preprocess the input dataframe by:
        - Dropping duplicates
        - Handling missing values
        - Encoding categorical variables using OneHotEncoder
        """
        # Drop duplicates
        df = df.drop_duplicates()

        # Handle missing values: impute categorical with mode and numeric with mean
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = self.imputer.fit_transform(df[[col]])
            else:
                df[col] = df[col].fillna(df[col].mean())

        # One-Hot Encoding for categorical columns
        categorical_columns = df.select_dtypes(include=["object"]).columns
        if len(categorical_columns) > 0:
            df_encoded = pd.DataFrame(self.ohe.fit_transform(df[categorical_columns]))
            df_encoded.columns = self.ohe.get_feature_names_out(categorical_columns)

            # Drop original categorical columns and concatenate encoded ones
            df = df.drop(columns=categorical_columns)
            df = pd.concat([df, df_encoded], axis=1)

        return df
