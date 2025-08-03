"""
Titanic Survival Prediction - Kaggle Grandmaster Solution
Author:
Description: Advanced machine learning pipeline with feature engineering and stacking ensemble
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import KNNImputer
from sklearn.base import BaseEstimator, TransformerMixin
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
import re

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

class TitanicFeatureEngineering(BaseEstimator, TransformerMixin):
    """
    Advanced feature engineering for Titanic dataset
    """
    
    def __init__(self):
        self.age_imputer = None
        self.fare_imputer = None
        self.title_mapping = {}
        
    def fit(self, X, y=None):
        """Fit the feature engineering pipeline"""
        X_copy = X.copy()
        
        # Fit age imputer based on Pclass and Sex
        age_features = ['Pclass', 'Sex', 'SibSp', 'Parch']
        X_age = pd.get_dummies(X_copy[age_features + ['Age']])
        mask = X_age['Age'].notna()
        
        if mask.sum() > 0:
            self.age_imputer = KNNImputer(n_neighbors=5)
            self.age_imputer.fit(X_age[mask])
        
        # Fit fare imputer
        fare_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']
        X_fare = pd.get_dummies(X_copy[fare_features + ['Fare']].fillna(X_copy[fare_features + ['Fare']].median()))
        
        self.fare_imputer = KNNImputer(n_neighbors=5)
        self.fare_imputer.fit(X_fare)
        
        # Learn title mapping
        titles = X_copy['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        title_counts = titles.value_counts()
        
        # Group rare titles
        common_titles = title_counts[title_counts >= 10].index.tolist()
        self.title_mapping = {title: title if title in common_titles else 'Rare' for title in title_counts.index}
        
        return self
    
    def transform(self, X):
        """Transform the dataset with advanced feature engineering"""
        X_copy = X.copy()
        
        # 1. Handle missing values intelligently
        X_copy = self._impute_missing_values(X_copy)
        
        # 2. Extract and engineer features
        X_copy = self._extract_title_features(X_copy)
        X_copy = self._extract_family_features(X_copy)
        X_copy = self._extract_cabin_features(X_copy)
        X_copy = self._extract_ticket_features(X_copy)
        X_copy = self._create_interaction_features(X_copy)
        X_copy = self._create_binned_features(X_copy)
        
        # 3. Handle categorical encoding
        X_copy = self._encode_categorical_features(X_copy)
        
        return X_copy
    
    def _impute_missing_values(self, X):
        """Intelligent missing value imputation"""
        # Age imputation using KNN based on other features
        if self.age_imputer is not None:
            age_features = ['Pclass', 'Sex', 'SibSp', 'Parch']
            X_age_features = pd.get_dummies(X[age_features + ['Age']])
            
            # Align columns with training data
            missing_cols = set(self.age_imputer.feature_names_in_) - set(X_age_features.columns)
            for col in missing_cols:
                X_age_features[col] = 0
            X_age_features = X_age_features[self.age_imputer.feature_names_in_]
            
            X_age_imputed = self.age_imputer.transform(X_age_features)
            X.loc[:, 'Age'] = X_age_imputed[:, -1]  # Age is the last column
        
        # Embarked: Most common port
        X['Embarked'].fillna('S', inplace=True)
        
        # Fare: Use KNN imputation
        if X['Fare'].isna().any():
            fare_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']
            X_fare_features = pd.get_dummies(X[fare_features + ['Fare']])
            
            # Align columns
            missing_cols = set(self.fare_imputer.feature_names_in_) - set(X_fare_features.columns)
            for col in missing_cols:
                X_fare_features[col] = 0
            X_fare_features = X_fare_features[self.fare_imputer.feature_names_in_]
            
            X_fare_imputed = self.fare_imputer.transform(X_fare_features)
            X.loc[:, 'Fare'] = X_fare_imputed[:, -1]
        
        return X
    
    def _extract_title_features(self, X):
        """Extract and engineer title features from names"""
        # Extract titles
        X['Title'] = X['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        
        # Apply learned mapping
        X['Title'] = X['Title'].map(self.title_mapping).fillna('Rare')
        
        # Title-based features
        X['Title_Mr'] = (X['Title'] == 'Mr').astype(int)
        X['Title_Mrs'] = (X['Title'] == 'Mrs').astype(int)
        X['Title_Miss'] = (X['Title'] == 'Miss').astype(int)
        X['Title_Master'] = (X['Title'] == 'Master').astype(int)
        X['Title_Rare'] = (X['Title'] == 'Rare').astype(int)
        
        # Name length (might indicate social status)
        X['Name_Length'] = X['Name'].str.len()
        
        return X
    
    def _extract_family_features(self, X):
        """Extract family-related features"""
        # Basic family size
        X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
        
        # Family size categories
        X['IsAlone'] = (X['FamilySize'] == 1).astype(int)
        X['SmallFamily'] = ((X['FamilySize'] >= 2) & (X['FamilySize'] <= 4)).astype(int)
        X['LargeFamily'] = (X['FamilySize'] >= 5).astype(int)
        
        # Family survival features (proxy for family composition)
        X['Has_Spouse'] = (X['SibSp'] > 0).astype(int)
        X['Has_Children'] = (X['Parch'] > 0).astype(int)
        X['Has_Parents'] = ((X['Age'] < 18) & (X['Parch'] > 0)).astype(int)
        
        # Extract surname for family groups
        X['Surname'] = X['Name'].str.extract('([A-Za-z]+),', expand=False)
        surname_counts = X['Surname'].value_counts()
        X['Family_Survival_Rate'] = X['Surname'].map(surname_counts)
        X['Large_Family_Group'] = (X['Family_Survival_Rate'] >= 3).astype(int)
        
        return X
    
    def _extract_cabin_features(self, X):
        """Extract cabin-related features"""
        # Has cabin information
        X['Has_Cabin'] = X['Cabin'].notna().astype(int)
        
        # Extract deck information
        X['Deck'] = X['Cabin'].str.extract('([A-Za-z])', expand=False)
        X['Deck'].fillna('Unknown', inplace=True)
        
        # Number of cabins
        X['Num_Cabins'] = X['Cabin'].str.split().str.len().fillna(0)
        
        # Cabin number (if available)
        X['Cabin_Number'] = X['Cabin'].str.extract('([0-9]+)', expand=False).astype(float)
        X['Has_Cabin_Number'] = X['Cabin_Number'].notna().astype(int)
        
        return X
    
    def _extract_ticket_features(self, X):
        """Extract ticket-related features"""
        # Ticket length
        X['Ticket_Length'] = X['Ticket'].str.len()
        
        # Ticket has letters
        X['Ticket_Has_Letters'] = X['Ticket'].str.contains('[A-Za-z]').astype(int)
        
        # Extract ticket number
        X['Ticket_Number'] = X['Ticket'].str.extract('([0-9]+)$', expand=False).astype(float)
        X['Has_Ticket_Number'] = X['Ticket_Number'].notna().astype(int)
        
        # Group tickets (same ticket = group booking)
        ticket_counts = X['Ticket'].value_counts()
        X['Ticket_Group_Size'] = X['Ticket'].map(ticket_counts)
        X['Is_Group_Ticket'] = (X['Ticket_Group_Size'] > 1).astype(int)
        
        return X
    
    def _create_interaction_features(self, X):
        """Create interaction features"""
        # Age-Class interaction
        X['Age_Class'] = X['Age'] * X['Pclass']
        
        # Fare per person
        X['Fare_Per_Person'] = X['Fare'] / X['FamilySize']
        
        # Age groups with class
        X['Age_Group'] = pd.cut(X['Age'], bins=[0, 12, 18, 35, 60, 100], 
                               labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
        
        # Class-specific features
        X['High_Fare_3rd_Class'] = ((X['Pclass'] == 3) & (X['Fare'] > X['Fare'].median())).astype(int)
        X['Low_Fare_1st_Class'] = ((X['Pclass'] == 1) & (X['Fare'] < X['Fare'].quantile(0.75))).astype(int)
        
        return X
    
    def _create_binned_features(self, X):
        """Create binned numerical features"""
        # Age bins
        X['Age_Bin'] = pd.cut(X['Age'], bins=5, labels=False)
        
        # Fare bins
        X['Fare_Bin'] = pd.qcut(X['Fare'], q=4, labels=False, duplicates='drop')
        
        return X
    
    def _encode_categorical_features(self, X):
        """Encode categorical features"""
        # Binary encoding for Sex
        X['Sex_male'] = (X['Sex'] == 'male').astype(int)
        X['Sex_female'] = (X['Sex'] == 'female').astype(int)
        
        # One-hot encoding for Embarked
        embarked_dummies = pd.get_dummies(X['Embarked'], prefix='Embarked')
        X = pd.concat([X, embarked_dummies], axis=1)
        
        # One-hot encoding for Deck
        deck_dummies = pd.get_dummies(X['Deck'], prefix='Deck')
        X = pd.concat([X, deck_dummies], axis=1)
        
        # One-hot encoding for Age_Group
        if 'Age_Group' in X.columns:
            age_group_dummies = pd.get_dummies(X['Age_Group'], prefix='AgeGroup')
            X = pd.concat([X, age_group_dummies], axis=1)
        
        return X

def perform_eda(train_df):
    """Perform comprehensive exploratory data analysis"""
    print("🔍 EXPLORATORY DATA ANALYSIS")
    print("=" * 50)
    
    # Basic information
    print(f"Dataset shape: {train_df.shape}")
    print(f"Missing values:\n{train_df.isnull().sum()}")
    print(f"\nSurvival rate: {train_df['Survived'].mean():.3f}")
    
    # Survival by key features
    print("\n📊 Survival Rates by Key Features:")
    print(f"By Sex:\n{train_df.groupby('Sex')['Survived'].agg(['count', 'mean'])}")
    print(f"\nBy Pclass:\n{train_df.groupby('Pclass')['Survived'].agg(['count', 'mean'])}")
    print(f"\nBy Embarked:\n{train_df.groupby('Embarked')['Survived'].agg(['count', 'mean'])}")
    
    # Age analysis
    print(f"\n👥 Age Statistics:")
    print(f"Age range: {train_df['Age'].min():.1f} - {train_df['Age'].max():.1f}")
    print(f"Mean age: {train_df['Age'].mean():.1f}")
    print(f"Missing ages: {train_df['Age'].isnull().sum()}")
    
    # Family analysis
    train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
    print(f"\n👨‍👩‍👧‍👦 Family Size vs Survival:")
    print(train_df.groupby('FamilySize')['Survived'].agg(['count', 'mean']))
    
    return train_df

class StackingEnsemble:
    """
    Advanced stacking ensemble with multiple base models
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.base_models = {}
        self.meta_model = None
        self.feature_importance_ = None
        
        # Initialize base models
        self.base_models['lr'] = LogisticRegression(
            random_state=random_state, 
            max_iter=1000,
            C=0.1
        )
        
        self.base_models['rf'] = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1
        )
        
        self.base_models['xgb'] = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            eval_metric='logloss'
        )
        
        self.base_models['lgb'] = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            verbose=-1
        )
        
        # Meta model
        self.meta_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=random_state,
            eval_metric='logloss'
        )
    
    def fit(self, X, y):
        """Train the stacking ensemble"""
        print("🚀 Training Stacking Ensemble...")
        
        # Cross-validation for base models
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        # Create meta-features
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            print(f"Training fold {fold + 1}/5...")
            
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            for i, (name, model) in enumerate(self.base_models.items()):
                # Train base model on fold
                model_fold = self._clone_model(model)
                model_fold.fit(X_train_fold, y_train_fold)
                
                # Predict on validation set
                pred_proba = model_fold.predict_proba(X_val_fold)[:, 1]
                meta_features[val_idx, i] = pred_proba
        
        # Train base models on full dataset
        print("Training base models on full dataset...")
        for name, model in self.base_models.items():
            model.fit(X, y)
        
        # Train meta model
        print("Training meta model...")
        self.meta_model.fit(meta_features, y)
        
        # Calculate feature importance (from best base model)
        self._calculate_feature_importance(X)
        
        print("✅ Ensemble training completed!")
    
    def predict(self, X):
        """Make predictions using the stacking ensemble"""
        # Get base model predictions
        base_predictions = np.zeros((X.shape[0], len(self.base_models)))
        
        for i, (name, model) in enumerate(self.base_models.items()):
            pred_proba = model.predict_proba(X)[:, 1]
            base_predictions[:, i] = pred_proba
        
        # Meta model prediction
        final_predictions = self.meta_model.predict(base_predictions)
        return final_predictions
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        # Get base model predictions
        base_predictions = np.zeros((X.shape[0], len(self.base_models)))
        
        for i, (name, model) in enumerate(self.base_models.items()):
            pred_proba = model.predict_proba(X)[:, 1]
            base_predictions[:, i] = pred_proba
        
        # Meta model prediction probabilities
        final_proba = self.meta_model.predict_proba(base_predictions)
        return final_proba
    
    def _clone_model(self, model):
        """Clone a model with same parameters"""
        return model.__class__(**model.get_params())
    
    def _calculate_feature_importance(self, X):
        """Calculate feature importance from the best performing base model"""
        # Use Random Forest feature importance as it's most interpretable
        if hasattr(self.base_models['rf'], 'feature_importances_'):
            self.feature_importance_ = pd.DataFrame({
                'feature': X.columns,
                'importance': self.base_models['rf'].feature_importances_
            }).sort_values('importance', ascending=False)

def evaluate_model(model, X_val, y_val, model_name="Model"):
    """Evaluate model performance"""
    predictions = model.predict(X_val)
    prob_predictions = model.predict_proba(X_val)[:, 1]
    
    accuracy = accuracy_score(y_val, predictions)
    
    print(f"\n📊 {model_name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_val, y_val, cv=5, scoring='accuracy')
    print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return accuracy, predictions

def main():
    """Main execution pipeline"""
    print("🚢 TITANIC SURVIVAL PREDICTION - GRANDMASTER SOLUTION")
    print("=" * 60)
    
    # Load data
    print("📁 Loading datasets...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    print(f"Training set: {train_df.shape}")
    print(f"Test set: {test_df.shape}")
    
    # Perform EDA
    train_df = perform_eda(train_df)
    
    # Combine datasets for feature engineering
    full_data = pd.concat([train_df.drop('Survived', axis=1), test_df], ignore_index=True)
    
    # Feature Engineering
    print("\n🔧 Performing Advanced Feature Engineering...")
    feature_engineer = TitanicFeatureEngineering()
    
    # Fit on training data only
    feature_engineer.fit(train_df.drop('Survived', axis=1))
    
    # Transform full dataset
    full_data_processed = feature_engineer.transform(full_data)
    
    # Split back into train and test
    train_processed = full_data_processed[:len(train_df)].copy()
    test_processed = full_data_processed[len(train_df):].copy()
    
    # Select features (remove original categorical and identifier columns)
    features_to_drop = ['PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 
                       'Title', 'Surname', 'Deck', 'Age_Group', 'Cabin_Number', 'Ticket_Number']
    
    # Keep only features that exist in the dataset
    features_to_drop = [f for f in features_to_drop if f in train_processed.columns]
    
    X = train_processed.drop(features_to_drop, axis=1)
    y = train_df['Survived']
    X_test = test_processed.drop(features_to_drop, axis=1)
    
    # Align test set columns with training set
    missing_cols = set(X.columns) - set(X_test.columns)
    for col in missing_cols:
        X_test[col] = 0
    X_test = X_test[X.columns]
    
    print(f"Final feature set: {X.shape[1]} features")
    print(f"Features: {list(X.columns)}")
    
    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), 
        columns=X_train.columns, 
        index=X_train.index
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val), 
        columns=X_val.columns, 
        index=X_val.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), 
        columns=X_test.columns, 
        index=X_test.index
    )
    
    # Train Stacking Ensemble
    ensemble = StackingEnsemble(random_state=RANDOM_STATE)
    ensemble.fit(X_train_scaled, y_train)
    
    # Evaluate ensemble
    accuracy, val_predictions = evaluate_model(ensemble, X_val_scaled, y_val, "Stacking Ensemble")
    
    # Feature importance
    if ensemble.feature_importance_ is not None:
        print("\n🎯 Top 15 Most Important Features:")
        print(ensemble.feature_importance_.head(15))
    
    # Make final predictions
    print("\n🔮 Making final predictions...")
    test_predictions = ensemble.predict(X_test_scaled)
    
    # Create submission
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': test_predictions
    })
    
    # Validate submission format
    print(f"\n📋 Submission Validation:")
    print(f"Shape: {submission.shape}")
    print(f"Columns: {list(submission.columns)}")
    print(f"Survived values: {sorted(submission['Survived'].unique())}")
    print(f"Missing values: {submission.isnull().sum().sum()}")
    
    # Save submission
    submission.to_csv('titanic_grandmaster_submission.csv', index=False)
    print("✅ Submission saved as 'titanic_grandmaster_submission.csv'")
    
    # Final summary
    print(f"\n🏆 FINAL RESULTS:")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Predicted Survival Rate: {submission['Survived'].mean():.3f}")
    print(f"Total Predictions: {len(submission)}")
    
    return submission, ensemble, feature_engineer

if __name__ == "__main__":
    submission, model, feature_eng = main()
