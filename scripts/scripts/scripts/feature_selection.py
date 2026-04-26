"""
Titanic Feature Selection Module
Uses multiple methods to select best features for prediction
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

class TitanicFeatureSelector:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.selected_features = []
        self.feature_importance = None
        
    def prepare_data(self, df, target='Survived'):
        """Prepare data for feature selection"""
        # Drop non-feature columns
        drop_cols = ['PassengerId', 'Name', 'Ticket']
        if target in df.columns:
            X = df.drop(drop_cols + [target], axis=1, errors='ignore')
            y = df[target]
        else:
            X = df.drop(drop_cols, axis=1, errors='ignore')
            y = None
        
        # Encode categorical variables
        label_encoders = {}
        for col in X.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        
        return X, y, label_encoders
    
    def correlation_analysis(self, X, threshold=0.8):
        """
        Remove highly correlated features
        """
        print("\n" + "="*40)
        print("CORRELATION ANALYSIS")
        print("="*40)
        
        # Calculate correlation matrix
        corr_matrix = X.corr()
        
        # Find highly correlated features
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_features = [column for column in upper.columns 
                              if any(abs(upper[column]) > threshold)]
        
        print(f"Features with correlation > {threshold}:")
        for col in high_corr_features:
            correlated_with = list(upper.index[abs(upper[col]) > threshold])
            print(f"  - {col} is highly correlated with {correlated_with}")
        
        # Drop highly correlated features
        X_reduced = X.drop(high_corr_features, axis=1, errors='ignore')
        print(f"\nDropped {len(high_corr_features)} highly correlated features")
        print(f"Remaining features: {X_reduced.shape[1]}")
        
        return X_reduced, high_corr_features
    
    def random_forest_importance(self, X, y, top_n=15):
        """
        Use Random Forest to determine feature importance
        """
        print("\n" + "="*40)
        print("RANDOM FOREST FEATURE IMPORTANCE")
        print("="*40)
        
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, 
                                    random_state=self.random_state,
                                    n_jobs=-1)
        rf.fit(X, y)
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop {top_n} most important features:")
        print(importance_df.head(top_n).to_string(index=False))
        
        # Store for later use
        self.feature_importance = importance_df
        
        # Select features above mean importance
        mean_importance = importance_df['importance'].mean()
        important_features = importance_df[importance_df['importance'] > mean_importance]['feature'].tolist()
        
        print(f"\nFeatures above mean importance ({mean_importance:.4f}): {len(important_features)}")
        
        return important_features
    
    def recursive_feature_elimination(self, X, y, n_features=15):
        """
        Perform Recursive Feature Elimination
        """
        print("\n" + "="*40)
        print("RECURSIVE FEATURE ELIMINATION")
        print("="*40)
        
        # Initialize estimator
        estimator = RandomForestClassifier(n_estimators=100, 
                                          random_state=self.random_state,
                                          n_jobs=-1)
        
        # Perform RFE
        rfe = RFE(estimator, n_features_to_select=n_features, step=1)
        rfe.fit(X, y)
        
        # Get selected features
        rfe_features = X.columns[rfe.support_].tolist()
        rfe_ranking = pd.DataFrame({
            'feature': X.columns,
            'rank': rfe.ranking_
        }).sort_values('rank')
        
        print(f"\nTop {n_features} features selected by RFE:")
        print(rfe_ranking.head(n_features).to_string(index=False))
        
        return rfe_features
    
    def select_features(self, train_df, target='Survived'):
        """Main feature selection pipeline"""
        print("="*60)
        print("STARTING FEATURE SELECTION")
        print("="*60)
        
        # Prepare data
        X, y, label_encoders = self.prepare_data(train_df, target)
        print(f"\nStarting features: {X.shape[1]}")
        print(f"Feature names: {list(X.columns)}")
        
        # 1. Correlation Analysis
        X_reduced, removed_corr = self.correlation_analysis(X)
        
        # 2. Random Forest Feature Importance
        rf_features = self.random_forest_importance(X, y)
        
        # 3. Recursive Feature Elimination (Optional)
        rfe_features = self.recursive_feature_elimination(X, y, n_features=12)
        
        # Combine methods - features that appear in at least 2 methods
        all_selected = []
        feature_sets = [rf_features, rfe_features]
        
        for feature in X.columns:
            count = sum(1 for feature_set in feature_sets if feature in feature_set)
            if count >= 1:  # Feature appears in at least 1 method
                all_selected.append(feature)
        
        # Remove duplicates and keep only features present in original data
        self.selected_features = list(set(all_selected))
        
        # Ensure target is not in features
        if target in self.selected_features:
            self.selected_features.remove(target)
        
        # Add justification
        self.create_feature_justification()
        
        print("\n" + "="*60)
        print(f"FINAL SELECTED FEATURES: {len(self.selected_features)}")
        print(self.selected_features)
        print("="*60)
        
        return self.selected_features
    
    def create_feature_justification(self):
        """Create justification for each selected feature"""
        print("\n" + "="*40)
        print("FEATURE SELECTION JUSTIFICATION")
        print("="*40)
        
        justifications = {
            'Pclass': 'Strong predictor - higher class passengers had priority access to lifeboats',
            'Sex': 'Critical - women and children were given priority during evacuation',
            'Age': 'Important demographic factor - children had higher survival priority',
            'SibSp': 'Family connections - having siblings/spouse aboard affected survival chances',
            'Parch': 'Family connections - parents/children relationships influenced survival',
            'Fare': 'Proxy for wealth and social status - correlated with survival',
            'Embarked': 'Port of embarkation may indicate travel purpose and demographics',
            'FamilySize': 'Family size affects evacuation coordination and survival chances',
            'IsAlone': 'Solo passengers may have different survival patterns than families',
            'Title': 'Social status indicator derived from names',
            'AgeGroup': 'Categorized age provides clearer patterns than continuous age',
            'IsChild': 'Binary child indicator captures the "women and children first" protocol',
            'Sex_Class': 'Interaction between gender and class reveals complex survival patterns',
            'FarePerPerson': 'Per-person fare is a better wealth indicator than total fare',
            'Has_Cabin': 'Cabin information availability may indicate passenger importance',
            'Deck': 'Deck location on ship affected access to lifeboats'
        }
        
        justification_df = []
        for feature in self.selected_features:
            if feature in justifications:
                justification_df.append({
                    'Feature': feature,
                    'Justification': justifications[feature]
                })
        
        if justification_df:
            just_df = pd.DataFrame(justification_df)
            print(just_df.to_string(index=False))
        
        # Save to file
        just_df.to_csv('../data/feature_justification.csv', index=False)
        print("\nJustification saved to '../data/feature_justification.csv'")

if __name__ == "__main__":
    # Load engineered data
    train_df = pd.read_csv('../data/train_engineered.csv')
    
    selector = TitanicFeatureSelector()
    selected_features = selector.select_features(train_df)
    
    # Save selected features
    pd.Series(selected_features).to_csv('../data/selected_features.csv', index=False)
    print("\nSelected features saved successfully!")
