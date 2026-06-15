import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders={}

    
    def load_data(self, filepath):
        print(f"Loding data from {filepath} ...")
        df = pd.read_csv(filepath)
        print("Data Loded!")
        return df


    def handling_missing_values(self, df):
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

        if df['TotalCharges'].isnull().sum()>0:
            df.fillna({'TotalCharges': df['TotalCharges'].median()}, inplace=True)

            print(f"Filled missing values")

        return df


    def features_engineering(self, df):
        # Tenure groups - FIX: Handle edge cases and NaN 
        # Use right=True to include right edge, and extend bins to cover all values
        df['tenure_group'] = pd.cut(df['tenure'],
        bins = [-1, 12, 24, 48, 73], 
        labels = [0, 1, 2, 3], 
        include_lowest = True)

        # Convert to float then int
        df['tenure_group'] = df['tenure_group'].astype(float).astype(int)

        # avg monthly charge per tenure
        df['avg_monthly_per_tenure'] = df['TotalCharges'] / (df['tenure'] + 1)

        # Multiple services
        service_cols = ['PhoneService', 'MultipleLines', 'InternetService', "onlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]

        # count services (handling 'No Internet service' property)
        df['num_services'] = 0

        for col in service_cols:
            if col in df.columns:
                # count as service if value is 'Yes' 
                df['num_services'] += (df[col] == 'Yes').astype(int)

                print(f" Created 3 new fetures")
                print(f"   -tenure_group: {df ['tenure_group'].unique()}")
                print(f"   -avg_monthly_per_tenure: min = {df['avg_monthly_per_tenure'].min():.2f}, max = {df['avg_monthly_per_tenure'].max():.2f}")
                print(f"   -num_services: min ={df['num_services'].min()}, max = {df['num_services'].max()}")

                return df


    def encode_features(self, df, is_traning=False):
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)

        #Binary encoding for Yes/No
        binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
        for col in binary_cols:
            df[col] = df[col].map({'Yes': 1, 'No': 0})

        # For Target variable 
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

        for col in categorical_cols:
            if is_traning:
                le = LabelEncoder()

                df[col] = le.fit_transform(df[col].astype(str))

                self.label_encoders[col] = le
                print(f"   -Encoded {col}: {len(le.classes_)} classes")

            else:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    df[col] = le.transform(df[col].astype(str))

            
        print(f" Encoded {len(binary_cols) + len(categorical_cols) + 1} categorical fetures")

        return df
    

    def scale_features(self, x_train, x_test=None, is_training=True):
        numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'avg_monthly_per_tenure']

        numerical_cols = [col for col in numerical_cols
        if col in x_train.columns]

        if is_training:
            x_train[numerical_cols] = self.scaler.fit_transform(x_train[numerical_cols])
            if x_test is not None:
                x_test[numerical_cols] = self.scaler.transform(x_test[numerical_cols])
            
        else:
            x_train[numerical_cols] = self.scaler.transform(x_train[numerical_cols])

        print("FEATURES SCALED!!")
        return x_train, x_test

    
    #  Preprocessing pipeline
    def prepare_data(self, df, test_size=0.2, random_state=42):
        """ Complete Preprocessing pipeline """
        print("\n " + "="*60)
        print("STARTING DATA PREPROCESSING PIPELINE")
        print("="*60)

        #handling missing values
        df = self.handling_missing_values(df)


        #feature engineering 
        df = self.features_engineering(df)

        #encoding
        df = self.encode_features(df, True)

        #split
        x = df.drop('Churn', axis=1)
        y = df['Churn']

        #traintest
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=y)

        print(f"\n Train-Test split: ")
        print(f"    Train set : {x_train.shape[0]} samples ({(1-test_size)*100:.0f}%)")
        print(f"    Test set : {x_test.shape[0]} samples ({test_size*100:.0f}%)")


        #scale features
        x_train, x_test = self.scale_features(x_train, x_test, is_training=True)



        

        print("\n " + "="*60)
        print(" PREPROCESSING COMPLETED")
        print("="*60)

        return x_train, x_test, y_train, y_test





if __name__ == "__main__":
    preprocessor = DataPreprocessor()

    df = preprocessor.load_data("Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    x_train, x_test, y_train, y_test = preprocessor.prepare_data(df)




    print(f"\n" + "="*60)
    print("VERIFICATION")
    print(f"\n" + "="*60)
    print(f" Training features: {x_train.shape}")
    print(f" Test features: {x_test.shape}")
    print(f" Training target: {y_train.shape}")
    print(f" Test target: {y_test.shape}")

    # Verify no object columns remain
    print(f"\n Final Data Types: ")
    print(x_train.dtypes.value_counts())

    # Sample of first few rows
    print(f"\n Sample Data (first 3 rows, first 5 columns: ")
    print(x_train.iloc[:3, :5])





