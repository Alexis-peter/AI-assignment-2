"""
Titanic Feature Engineering Module
Creates new features from existing data
"""

import pandas as pd
import numpy as np

class TitanicFeatureEngineer:
    def __init__(self):
        self.engineered_features = []
        
    def create_family_features(self, df):
        """
        Create family-related features:
        - FamilySize: Total family members
        - IsAlone: Whether passenger is traveling alone
        - FamilyType: Categorized family groups
        """
        df_fe = df.copy()
        
        # Family size (including self)
        df_fe['FamilySize'] = df_fe['SibSp'] + df_fe['Parch'] + 1
        
        # Is alone indicator
        df_fe['IsAlone'] = (df_fe['FamilySize'] == 1).astype(int)
        
        # Family type categories
        df_fe['FamilyType'] = pd.cut(df_fe['FamilySize'], 
                                      bins=[0, 1, 2, 4, 7, 20],
                                      labels=['Alone', 'Small', 'Medium', 'Large', 'Very Large'])
        
        self.engineered_features.extend(['FamilySize', 'IsAlone', 'FamilyType'])
        print(f"Created family features: FamilySize, IsAlone, FamilyType")
        
        return df_fe
    
    def create_title_feature(self, df):
        """
        Extract title from Name
        Group rare titles into categories
        """
        df_fe = df.copy()
        
        # Extract title from name
        df_fe['Title'] = df_fe['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        
        # Group rare titles
        title_mapping = {
            'Mr': 'Mr',
            'Mrs': 'Mrs',
            'Miss': 'Miss',
            'Master': 'Master',
            'Don': 'Rare',
            'Rev': 'Rare',
            'Dr': 'Dr',
            'Mme': 'Mrs',
            'Ms': 'Miss',
            'Major': 'Military',
            'Lady': 'Rare',
            'Sir': 'Rare',
            'Mlle': 'Miss',
            'Col': 'Military',
            'Capt': 'Military',
            'Countess': 'Rare',
            'Jonkheer': 'Rare'
        }
        
        df_fe['Title'] = df_fe['Title'].map(title_mapping)
        df_fe['Title'] = df_fe['Title'].fillna('Rare')
        
        self.engineered_features.append('Title')
        print("Created Title feature")
        
        return df_fe
    
    def create_age_features(self, df):
        """
        Create age-related features:
        - AgeGroup: Categorized age groups
        - IsChild: Whether passenger is a child
        """
        df_fe = df.copy()
        
        # Age groups
        df_fe['AgeGroup'] = pd.cut(df_fe['Age'],
                                    bins=[0, 12, 18, 30, 50, 100],
                                    labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])
        
        # Is child indicator
        df_fe['IsChild'] = (df_fe['Age'] <= 12).astype(int)
        
        self.engineered_features.extend(['AgeGroup', 'IsChild'])
        print("Created age features: AgeGroup, IsChild")
        
        return df_fe
    
    def create_fare_features(self, df):
        """
        Create fare-related features:
        - FarePerPerson: Fare divided by family size
        - FareCategory: Categorized fare ranges
        """
        df_fe = df.copy()
        
        # Fare per person
        df_fe['FarePerPerson'] = df_fe['Fare'] / df_fe['FamilySize']
        
        # Fare categories
        df_fe['FareCategory'] = pd.qcut(df_fe['Fare'], 
                                         q=4, 
                                         labels=['Low', 'Medium', 'High', 'Very High'])
        
        self.engineered_features.extend(['FarePerPerson', 'FareCategory'])
        print("Created fare features: FarePerPerson, FareCategory")
        
        return df_fe
    
    def create_interaction_features(self, df):
        """
        Create interaction features between existing features
        """
        df_fe = df.copy()
        
        # Sex and Pclass interaction (higher class women had higher survival)
        df_fe['Sex_Class'] = df_fe['Sex'].astype(str) + '_' + df_fe['Pclass'].astype(str)
        
        # Age and Class interaction
        df_fe['Age_Class'] = df_fe['AgeGroup'].astype(str) + '_' + df_fe['Pclass'].astype(str)
        
        self.engineered_features.extend(['Sex_Class', 'Age_Class'])
        print("Created interaction features: Sex_Class, Age_Class")
        
        return df_fe
    
    def engineer_all(self, train_df, test_df):
        """Main feature engineering pipeline"""
        print("="*60)
        print("STARTING FEATURE ENGINEERING")
        print("="*60)
        
        # Combine for consistent engineering
        train_len = len(train_df)
        combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)
        
        print(f"\nCombined dataset shape: {combined.shape}")
        
        # Apply feature engineering steps
        combined = self.create_family_features(combined)
        combined = self.create_title_feature(combined)
        combined = self.create_age_features(combined)
        combined = self.create_fare_features(combined)
        combined = self.create_interaction_features(combined)
        
        # Split back
        train_eng = combined[:train_len]
        test_eng = combined[train_len:]
        
        print("\n" + "="*60)
        print(f"FEATURE ENGINEERING COMPLETED")
        print(f"Total engineered features: {len(self.engineered_features)}")
        print(f"Engineered features: {self.engineered_features}")
        print("="*60)
        
        return train_eng, test_eng

if __name__ == "__main__":
    # Load cleaned data
    train_df = pd.read_csv('../data/train_clean.csv')
    test_df = pd.read_csv('../data/test_clean.csv')
    
    engineer = TitanicFeatureEngineer()
    train_eng, test_eng = engineer.engineer_all(train_df, test_df)
    
    # Save engineered datasets
    train_eng.to_csv('../data/train_engineered.csv', index=False)
    test_eng.to_csv('../data/test_engineered.csv', index=False)
    print("\nEngineered datasets saved successfully!")
