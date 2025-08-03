import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🚢 Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🚢 Titanic Survival Prediction Dashboard</h1>', unsafe_allow_html=True)

# Load data function
@st.cache_data
def load_data():
    """Load and return the Titanic datasets"""
    try:
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
        return train_df, test_df
    except FileNotFoundError:
        st.error("Please make sure train.csv and test.csv are in the same directory as this script.")
        return None, None

# Feature engineering function
def feature_engineering(df, is_train=True):
    """Apply feature engineering to the dataset"""
    df = df.copy()
    
    # Fill missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    
    # Create new features
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 60, 100], labels=['Child', 'Teen', 'Adult', 'Elder'])
    df['FareGroup'] = pd.qcut(df['Fare'].rank(method='first'), q=4, labels=['Low', 'Medium', 'High', 'Very High'])
    
    # Extract title from name
    df['Title'] = df['Name'].str.extract('([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    # Encode categorical variables
    le = LabelEncoder()
    df['Sex_encoded'] = le.fit_transform(df['Sex'])
    df['Embarked_encoded'] = le.fit_transform(df['Embarked'].astype(str))
    df['Title_encoded'] = le.fit_transform(df['Title'])
    df['AgeGroup_encoded'] = le.fit_transform(df['AgeGroup'].astype(str))
    df['FareGroup_encoded'] = le.fit_transform(df['FareGroup'].astype(str))
    
    return df

# Sidebar
st.sidebar.title("🎛️ Navigation")
page = st.sidebar.selectbox("Choose a page:", [
    "📊 Data Overview", 
    "📈 Data Analysis", 
    "🤖 Machine Learning", 
    "🔮 Make Predictions"
])

# Load data
train_df, test_df = load_data()

if train_df is not None and test_df is not None:
    
    # Apply feature engineering
    train_processed = feature_engineering(train_df, is_train=True)
    test_processed = feature_engineering(test_df, is_train=False)
    
    if page == "📊 Data Overview":
        st.header("📊 Dataset Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Training Data")
            st.write(f"**Shape:** {train_df.shape[0]} rows × {train_df.shape[1]} columns")
            st.dataframe(train_df.head())
            
        with col2:
            st.subheader("📋 Test Data")
            st.write(f"**Shape:** {test_df.shape[0]} rows × {test_df.shape[1]} columns")
            st.dataframe(test_df.head())
        
        # Data quality overview
        st.subheader("🔍 Data Quality Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            survival_rate = train_df['Survived'].mean() * 100
            st.metric("Overall Survival Rate", f"{survival_rate:.1f}%")
        
        with col2:
            missing_age = train_df['Age'].isnull().sum()
            st.metric("Missing Ages", f"{missing_age}")
        
        with col3:
            missing_cabin = train_df['Cabin'].isnull().sum()
            st.metric("Missing Cabins", f"{missing_cabin}")
        
        with col4:
            avg_age = train_df['Age'].mean()
            st.metric("Average Age", f"{avg_age:.1f} years")
        
        # Missing values heatmap
        st.subheader("🌡️ Missing Values Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(train_df.isnull(), cbar=True, yticklabels=False, cmap='viridis', ax=ax)
        plt.title('Missing Values in Training Data')
        st.pyplot(fig)
        
        # Dataset info
        st.subheader("📝 Column Descriptions")
        descriptions = {
            'PassengerId': 'Unique identifier for each passenger',
            'Survived': 'Survival (0 = No, 1 = Yes) - TARGET VARIABLE',
            'Pclass': 'Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)',
            'Name': 'Passenger name',
            'Sex': 'Gender (male/female)',
            'Age': 'Age in years',
            'SibSp': 'Number of siblings/spouses aboard',
            'Parch': 'Number of parents/children aboard',
            'Ticket': 'Ticket number',
            'Fare': 'Passenger fare',
            'Cabin': 'Cabin number',
            'Embarked': 'Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)'
        }
        
        for col, desc in descriptions.items():
            st.write(f"**{col}:** {desc}")
    
    elif page == "📈 Data Analysis":
        st.header("📈 Exploratory Data Analysis")
        
        # Survival by different factors
        st.subheader("🎯 Survival Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Survival by Gender
            survival_gender = train_df.groupby('Sex')['Survived'].agg(['count', 'sum', 'mean']).reset_index()
            survival_gender['survival_rate'] = survival_gender['mean'] * 100
            
            fig = px.bar(survival_gender, x='Sex', y='survival_rate', 
                        title='Survival Rate by Gender',
                        labels={'survival_rate': 'Survival Rate (%)'})
            fig.update_traces(text=survival_gender['survival_rate'].round(1), textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Survival by Class
            survival_class = train_df.groupby('Pclass')['Survived'].agg(['count', 'sum', 'mean']).reset_index()
            survival_class['survival_rate'] = survival_class['mean'] * 100
            survival_class['Pclass'] = survival_class['Pclass'].astype(str)
            
            fig = px.bar(survival_class, x='Pclass', y='survival_rate',
                        title='Survival Rate by Passenger Class',
                        labels={'survival_rate': 'Survival Rate (%)', 'Pclass': 'Passenger Class'})
            fig.update_traces(text=survival_class['survival_rate'].round(1), textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        
        # Age distribution
        st.subheader("👥 Age Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(train_df, x='Age', color='Survived', 
                             title='Age Distribution by Survival',
                             labels={'Survived': 'Survived'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Age vs Survival box plot
            fig = px.box(train_df, x='Survived', y='Age',
                        title='Age Distribution by Survival Status')
            st.plotly_chart(fig, use_container_width=True)
        
        # Fare analysis
        st.subheader("💰 Fare Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(train_df, x='Fare', color='Survived',
                             title='Fare Distribution by Survival')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(train_df, x='Pclass', y='Fare', color='Survived',
                        title='Fare by Class and Survival')
            st.plotly_chart(fig, use_container_width=True)
        
        # Family size analysis
        st.subheader("👨‍👩‍👧‍👦 Family Analysis")
        train_processed['FamilySize'] = train_processed['SibSp'] + train_processed['Parch'] + 1
        family_survival = train_processed.groupby('FamilySize')['Survived'].mean().reset_index()
        
        fig = px.line(family_survival, x='FamilySize', y='Survived',
                     title='Survival Rate by Family Size',
                     labels={'Survived': 'Survival Rate'})
        fig.update_traces(mode='markers+lines')
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("🔗 Feature Correlations")
        numeric_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
        corr_matrix = train_df[numeric_cols].corr()
        
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                       title="Correlation Matrix of Numeric Features")
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "🤖 Machine Learning":
        st.header("🤖 Machine Learning Models")
        
        # Feature selection
        feature_cols = ['Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Parch', 'Fare', 
                       'Embarked_encoded', 'FamilySize', 'IsAlone', 'Title_encoded']
        
        X = train_processed[feature_cols]
        y = train_processed['Survived']
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Model selection
        st.subheader("🎛️ Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_choice = st.selectbox("Choose Model:", [
                "Random Forest",
                "Gradient Boosting", 
                "Logistic Regression",
                "Support Vector Machine"
            ])
        
        with col2:
            use_scaling = st.checkbox("Use Feature Scaling", value=True)
        
        # Model training
        if st.button("🚀 Train Model"):
            with st.spinner("Training model..."):
                
                # Select data based on scaling preference
                X_train_final = X_train_scaled if use_scaling else X_train
                X_val_final = X_val_scaled if use_scaling else X_val
                
                # Initialize model
                if model_choice == "Random Forest":
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                elif model_choice == "Gradient Boosting":
                    model = GradientBoostingClassifier(random_state=42)
                elif model_choice == "Logistic Regression":
                    model = LogisticRegression(random_state=42, max_iter=1000)
                else:  # SVM
                    model = SVC(random_state=42, probability=True)
                
                # Train model
                model.fit(X_train_final, y_train)
                
                # Make predictions
                y_pred = model.predict(X_val_final)
                y_pred_proba = model.predict_proba(X_val_final)[:, 1]
                
                # Calculate metrics
                accuracy = accuracy_score(y_val, y_pred)
                
                # Cross validation
                cv_scores = cross_val_score(model, X_train_final, y_train, cv=5)
                
                # Store model in session state
                st.session_state['trained_model'] = model
                st.session_state['scaler'] = scaler if use_scaling else None
                st.session_state['feature_cols'] = feature_cols
                st.session_state['use_scaling'] = use_scaling
                
                # Display results
                st.success("Model trained successfully!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Validation Accuracy", f"{accuracy:.3f}")
                
                with col2:
                    st.metric("CV Mean Score", f"{cv_scores.mean():.3f}")
                
                with col3:
                    st.metric("CV Std Score", f"{cv_scores.std():.3f}")
                
                # Classification report
                st.subheader("📊 Classification Report")
                report = classification_report(y_val, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)
                
                # Confusion Matrix
                st.subheader("🎯 Confusion Matrix")
                cm = confusion_matrix(y_val, y_pred)
                
                fig = px.imshow(cm, text_auto=True, aspect="auto",
                               labels=dict(x="Predicted", y="Actual"),
                               x=['Did not survive', 'Survived'],
                               y=['Did not survive', 'Survived'],
                               title="Confusion Matrix")
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance (for tree-based models)
                if model_choice in ["Random Forest", "Gradient Boosting"]:
                    st.subheader("🎯 Feature Importance")
                    importance_df = pd.DataFrame({
                        'feature': feature_cols,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    fig = px.bar(importance_df, x='importance', y='feature',
                               orientation='h', title='Feature Importance')
                    st.plotly_chart(fig, use_container_width=True)
    
    elif page == "🔮 Make Predictions":
        st.header("🔮 Make Predictions on Test Data")
        
        if 'trained_model' not in st.session_state:
            st.warning("⚠️ Please train a model first in the Machine Learning section!")
        else:
            model = st.session_state['trained_model']
            scaler = st.session_state.get('scaler')
            feature_cols = st.session_state['feature_cols']
            use_scaling = st.session_state['use_scaling']
            
            st.success("✅ Using previously trained model")
            
            # Prepare test data
            X_test = test_processed[feature_cols]
            
            if use_scaling and scaler is not None:
                X_test_final = scaler.transform(X_test)
            else:
                X_test_final = X_test
            
            # Make predictions
            predictions = model.predict(X_test_final)
            probabilities = model.predict_proba(X_test_final)[:, 1]
            
            # Create submission dataframe
            submission_df = pd.DataFrame({
                'PassengerId': test_df['PassengerId'],
                'Survived': predictions
            })
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Prediction Summary")
                survival_count = predictions.sum()
                total_count = len(predictions)
                st.metric("Predicted Survivors", f"{survival_count} / {total_count}")
                st.metric("Predicted Survival Rate", f"{(survival_count/total_count)*100:.1f}%")
            
            with col2:
                st.subheader("📈 Prediction Distribution")
                pred_dist = pd.Series(predictions).value_counts()
                fig = px.pie(values=pred_dist.values, names=['Did not survive', 'Survived'],
                           title='Prediction Distribution')
                st.plotly_chart(fig, use_container_width=True)
            
            # Show predictions with probabilities
            st.subheader("🔍 Detailed Predictions")
            
            display_df = test_df[['PassengerId', 'Name', 'Sex', 'Age', 'Pclass']].copy()
            display_df['Predicted_Survival'] = predictions
            display_df['Survival_Probability'] = probabilities
            display_df['Prediction_Confidence'] = np.where(
                probabilities > 0.5, 
                probabilities, 
                1 - probabilities
            )
            
            # Add filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                gender_filter = st.selectbox("Filter by Gender:", ['All', 'male', 'female'])
            
            with col2:
                class_filter = st.selectbox("Filter by Class:", ['All', 1, 2, 3])
            
            with col3:
                prediction_filter = st.selectbox("Filter by Prediction:", ['All', 'Survived', 'Did not survive'])
            
            # Apply filters
            filtered_df = display_df.copy()
            
            if gender_filter != 'All':
                filtered_df = filtered_df[filtered_df['Sex'] == gender_filter]
            
            if class_filter != 'All':
                filtered_df = filtered_df[filtered_df['Pclass'] == class_filter]
            
            if prediction_filter != 'All':
                pred_value = 1 if prediction_filter == 'Survived' else 0
                filtered_df = filtered_df[filtered_df['Predicted_Survival'] == pred_value]
            
            st.dataframe(filtered_df, use_container_width=True)
            
            # Validate and prepare submission file
            st.subheader("💾 Download Submission")
            
            # Validation checks
            st.write("**📋 Submission Validation:**")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                row_count = len(submission_df)
                row_check = row_count == 418
                st.metric(
                    "Row Count", 
                    f"{row_count}/418", 
                    delta="✅ Correct" if row_check else "❌ Error"
                )
            
            with col2:
                col_count = len(submission_df.columns)
                col_check = col_count == 2
                st.metric(
                    "Column Count", 
                    f"{col_count}/2", 
                    delta="✅ Correct" if col_check else "❌ Error"
                )
            
            with col3:
                col_names_check = list(submission_df.columns) == ['PassengerId', 'Survived']
                st.metric(
                    "Column Names", 
                    "✅ Correct" if col_names_check else "❌ Error"
                )
            
            with col4:
                binary_check = submission_df['Survived'].isin([0, 1]).all()
                st.metric(
                    "Binary Values", 
                    "✅ Correct" if binary_check else "❌ Error"
                )
            
            # Overall validation status
            all_checks_pass = row_check and col_check and col_names_check and binary_check
            
            if all_checks_pass:
                st.success("🎉 Submission file format is correct and ready for Kaggle!")
                
                # Show submission preview
                st.write("**📋 Submission Preview (First 10 rows):**")
                st.code(f"""PassengerId,Survived
{submission_df.iloc[0]['PassengerId']},{submission_df.iloc[0]['Survived']}
{submission_df.iloc[1]['PassengerId']},{submission_df.iloc[1]['Survived']}
{submission_df.iloc[2]['PassengerId']},{submission_df.iloc[2]['Survived']}
{submission_df.iloc[3]['PassengerId']},{submission_df.iloc[3]['Survived']}
{submission_df.iloc[4]['PassengerId']},{submission_df.iloc[4]['Survived']}
{submission_df.iloc[5]['PassengerId']},{submission_df.iloc[5]['Survived']}
{submission_df.iloc[6]['PassengerId']},{submission_df.iloc[6]['Survived']}
{submission_df.iloc[7]['PassengerId']},{submission_df.iloc[7]['Survived']}
{submission_df.iloc[8]['PassengerId']},{submission_df.iloc[8]['Survived']}
{submission_df.iloc[9]['PassengerId']},{submission_df.iloc[9]['Survived']}
Etc.""")
                
                # Generate CSV for download
                csv = submission_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Kaggle Submission CSV",
                    data=csv,
                    file_name="titanic_submission.csv",
                    mime="text/csv",
                    help="Download the properly formatted submission file for Kaggle"
                )
                
                st.info("💡 Upload this CSV file to Kaggle to submit your predictions!")
                
            else:
                st.error("❌ Submission file format validation failed! Please check the requirements.")
            
            # Show full submission format requirements
            with st.expander("📋 Kaggle Submission Requirements"):
                st.markdown("""
                **Submission File Format Requirements:**
                
                ✅ **Exactly 418 entries plus header row**  
                ✅ **Exactly 2 columns: PassengerId and Survived**  
                ✅ **PassengerId can be in any order**  
                ✅ **Survived contains binary predictions: 1 for survived, 0 for deceased**  
                
                **Example format:**
                ```
                PassengerId,Survived
                892,0
                893,1
                894,0
                Etc.
                ```
                
                **Current submission summary:**
                - Total rows: {row_count} (including header)
                - Data rows: {len(submission_df)}
                - Columns: {list(submission_df.columns)}
                - Survived values: {sorted(submission_df['Survived'].unique())}
                """.format(
                    row_count=len(submission_df) + 1,
                    len=len
                ))
                
                st.dataframe(submission_df.head(15), use_container_width=True)

else:
    st.error("❌ Could not load the dataset files. Please make sure train.csv and test.csv are available.")
    st.info("💡 Make sure to upload the Titanic dataset files or place them in the same directory as this script.")