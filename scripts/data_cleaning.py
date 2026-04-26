"""
Titanic Data Cleaning Module
Handles missing values, outliers, and data type corrections
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

class TitanicDataCleaner:
    def __init__(self, train_path='../data/train.csv', test_path='../data/test.csv'):
        self.train_path = train_path
        self.test_path = test_path
        self.train_df = None
        self.test_df = None
        
    def load_data(self):
        """Load training and test datasets"""
        self.train_df = pd.read_csv(self.train_path)
        self.test_df = pd.read_csv(self.test_path)
        print(f"Training data shape: {self.train_df.shape}")
        print(f"Test data shape: {self.test_df.shape}")
        return self.train_df, self.test_df
    
    def check_missing_values(self, df, dataset_name=""):
        """Check and report missing values"""
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Percentage': missing_pct
        })
        print(f"\n{d} Missing Values Analysis:")
        print(missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False))
        return missing_df
    
    def clean_age(self, df):
        """
        Clean Age feature:
        - Fill missing ages using group medians based on Pclass and Sex
        - This preserves the relationship between age, class, and gender
        """
        df_clean = df.copy()
        
        # Calculate median age for each group (Pclass and Sex)
        age_medians = df_clean.groupby(['Pclass', 'Sex'])['Age'].transform('median')
        df_clean['Age'] = df_clean['Age'].fillna(age_medians)
        
        # For any remaining NaN (if group has no data), fill with overall median
        df_clean['Age'] = df_clean['Age'].fillna(df_clean['Age'].median())
        
        print(f"Age missing values after cleaning: {df_clean['Age'].isnull().sum()}")
        return df_clean
    
    def clean_embarked(self, df):
        """Clean Embarked feature by filling with mode"""
        df_clean = df.copy()
        # Fill missing Embarked with most common port
        embarked_mode = df_clean['Embarked'].mode()[0]
        df_clean['Embarked'] = df_clean['Embarked'].fillna(embarked_mode)
        print(f"Embarked missing values after cleaning: {df_clean['Embarked'].isnull().sum()}")
        return df_clean
    
    def clean_fare(self, df):
        """Clean Fare feature by filling with median based on Pclass"""
        df_clean = df.copy()
        # Fill missing Fare with median of Pclass
        fare_medians = df_clean.groupby('Pclass')['Fare'].transform('median')
        df_clean['Fare'] = df_clean['Fare'].fillna(fare_medians)
        print(f"Fare missing values after cleaning: {df_clean['Fare'].isnull().sum()}")
        return df_clean
    
    def clean_cabin(self, df):
        """
        Clean Cabin feature:
        - Extract deck letter from cabin
        - Create categorical feature for deck
        - Handle missing values
        """
        df_clean = df.copy()
        
        # Extract first letter of cabin (deck)
        df_clean['Deck'] = df_clean['Cabin'].str[0]
        
        # Fill missing Deck with 'U' for Unknown
        df_clean['Deck'] = df_clean['Deck'].fillna('U')
        
        # Keep original Cabin for potential feature engineering
        # but mark whether cabin information exists
        df_clean['Has_Cabin'] = df_clean['Cabin'].notna().astype(int)
        
        print(f"Deck value counts:\n{df_clean['Deck'].value_counts()}")
        return df_clean
    
    def handle_outliers(self, df, column, method='iqr'):
        """
        Handle outliers using IQR method
        """
        df_clean = df.copy()
        
        if method == 'iqr':
            Q1 = df_clean[column].quantile(0.25)
            Q3 = df_clean[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df_clean[(df_clean[column] < lower_bound) | 
                               (df_clean[column] > upper_bound)]
            
            if len(outliers) > 0:
                print(f"Found {len(outliers)} outliers in {column}")
                # Cap outliers instead of removing
                df_clean[column] = df_clean[column].clip(lower_bound, upper_bound)
                
        return df_clean
    
    def clean_all(self):
        """Main cleaning pipeline"""
        print("="*60)
        print("STARTING DATA CLEANING PROCESS")
        print("="*60)
        
        # Load data
        self.train_df, self.test_df = self.load_data()
        
        # Combine datasets for consistent cleaning
        train_len = len(self.train_df)
        combined = pd.concat([self.train_df, self.test_df], axis=0, ignore_index=True)
        
        print(f"\nCombined dataset shape: {combined.shape}")
        
        # Check missing values before cleaning
        self.missing_before = self.check_missing_values(combined, "BEFORE")
        
        # Apply cleaning steps
        print("\n" + "="*40)
        print("APPLYING CLEANING STEPS")
        print("="*40)
        
        combined = self.clean_age(combined)
        combined = self.clean_embarked(combined)
        combined = self.clean_fare(combined)
        combined = self.clean_cabin(combined)
        
        # Handle outliers in Fare
        combined = self.handle_outliers(combined, 'Fare', method='iqr')
        
        # Check missing values after cleaning
        self.missing_after = self.check_missing_values(combined, "AFTER")
        
        # Drop Cabin column (we have Deck and Has_Cabin instead)
        combined = combined.drop('Cabin', axis=1)
        
        # Split back into train and test
        train_clean = combined[:train_len]
        test_clean = combined[train_len:]
        
        print("\n" + "="*60)
        print("DATA CLEANING COMPLETED")
        print("="*60)
        
        return train_clean, test_clean

if __name__ == "__main__":
    cleaner = TitanicDataCleaner()
    train_clean, test_clean = cleaner.clean_all()
    
    # Save cleaned datasets
    train_clean.to_csv('../data/train_clean.csv', index=False)
    test_clean.to_csv('../data/test_clean.csv', index=False)
    print("\nCleaned datasets saved successfully!")
