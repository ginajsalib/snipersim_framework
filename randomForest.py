# Random Forest Configuration Prediction Model
# Google Colab Notebook for predicting best_config labels

# Install required packages (uncomment if running in Colab)
# !pip install pandas scikit-learn matplotlib seaborn

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
TARGET_COLUMNS = ['size0', 'size1']  # Two separate target columns

# Upload your CSV file (uncomment for Colab file upload)
# from google.colab import files
# uploaded = files.upload()
# csv_filename = list(uploaded.keys())[0]

# For local testing, specify your CSV file path
csv_filename = 'your_data.csv'  # Replace with your actual file path

print("🚀 Random Forest Configuration Prediction Model")
print("=" * 50)

# Load the dataset
try:
    df = pd.read_csv(csv_filename)
    print(f"✅ Data loaded successfully! Shape: {df.shape}")
    print(f"📊 Columns: {list(df.columns)}")
    print("\n🔍 First few rows:")
    print(df.head())
except FileNotFoundError:
    print("❌ CSV file not found. Please check the file path.")
    print("💡 If using Colab, uncomment the file upload section above.")

# Data exploration
def explore_data(df):
    print("\n" + "="*50)
    print("📈 DATA EXPLORATION")
    print("="*50)

    print(f"Dataset shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")

    if any(col in df.columns for col in TARGET_COLUMNS):
        print(f"\n🎯 Target variables distribution:")
        for target_col in TARGET_COLUMNS:
            if target_col in df.columns:
                print(f"\n{target_col}:")
                target_counts = df[target_col].value_counts().head(10)
                print(target_counts)

        # Plot target distributions
        fig, axes = plt.subplots(1, len(TARGET_COLUMNS), figsize=(15, 6))
        if len(TARGET_COLUMNS) == 1:
            axes = [axes]

        for i, target_col in enumerate(TARGET_COLUMNS):
            if target_col in df.columns:
                df[target_col].value_counts().head(10).plot(kind='bar', ax=axes[i])
                axes[i].set_title(f'Distribution of {target_col}')
                axes[i].set_xlabel('Value')
                axes[i].set_ylabel('Count')
                axes[i].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()
    else:
        print(f"❌ Target columns {TARGET_COLUMNS} not found in the dataset!")
        print(f"Available columns: {list(df.columns)}")

        # Check if old format exists and offer to split it
        if 'best_config' in df.columns:
            print("\n💡 Found 'best_config' column. Would you like to split it into size1 and size2?")
            print("Uncomment the following code to auto-split:")
            print("# df[['size1', 'size2']] = df['best_config'].str.split('-', expand=True)")
            print("# df = df.drop('best_config', axis=1)")

# Preprocessing function
def preprocess_data(df, remove_columns=None, enable_preprocessing=True):
    print("\n" + "="*50)
    print("🔧 DATA PREPROCESSING")
    print("="*50)

    df_processed = df.copy()

    if not enable_preprocessing:
        print("⚠️ Preprocessing disabled - using raw data")
        return df_processed

    # Remove unnecessary columns if specified
    if remove_columns:
        print(f"🗑️ Removing columns: {remove_columns}")
        df_processed = df_processed.drop(columns=remove_columns, errors='ignore')

    # Remove columns with too many missing values (>50%)
    missing_threshold = 0.5
    high_missing_cols = df_processed.columns[df_processed.isnull().mean() > missing_threshold].tolist()
    if high_missing_cols:
        print(f"🗑️ Removing high-missing columns (>{missing_threshold*100}%): {high_missing_cols}")
        df_processed = df_processed.drop(columns=high_missing_cols)

    # Remove constant columns (no variance)
    constant_cols = [col for col in df_processed.columns if df_processed[col].nunique() <= 1]
    if constant_cols:
        print(f"🗑️ Removing constant columns: {constant_cols}")
        df_processed = df_processed.drop(columns=constant_cols)

    # Handle missing values
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    categorical_cols = df_processed.select_dtypes(include=['object']).columns

    # Fill numeric missing values with median
    for col in numeric_cols:
        if df_processed[col].isnull().sum() > 0:
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
            print(f"📊 Filled missing values in '{col}' with median")

    # Fill categorical missing values with mode
    for col in categorical_cols:
        if col != 'best_config' and df_processed[col].isnull().sum() > 0:
            df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
            print(f"📝 Filled missing values in '{col}' with mode")

    print(f"✅ Preprocessing complete! New shape: {df_processed.shape}")
    return df_processed

# Feature preparation
def prepare_features(df):
    print("\n" + "="*50)
    print("🎯 FEATURE PREPARATION")
    print("="*50)

    # Check if target columns exist
    missing_targets = [col for col in TARGET_COLUMNS if col not in df.columns]
    if missing_targets:
        raise ValueError(f"Target columns {missing_targets} not found!")

    # Separate features and targets
    X = df.drop(TARGET_COLUMNS, axis=1)
    y = df[TARGET_COLUMNS].copy()

    # Convert targets to appropriate data types
    for col in TARGET_COLUMNS:
        # Try to convert to numeric, if fails keep as categorical
        try:
            y[col] = pd.to_numeric(y[col])
            print(f"🔢 Target '{col}' treated as numeric (regression)")
        except (ValueError, TypeError):
            print(f"📝 Target '{col}' treated as categorical (classification)")

    # Encode categorical variables in features
    categorical_cols = X.select_dtypes(include=['object']).columns
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        print(f"🔄 Encoded categorical column: {col}")

    print(f"✅ Features prepared! Shape: {X.shape}")
    print(f"🎯 Target 1 ({TARGET_COLUMNS[0]}): {y[TARGET_COLUMNS[0]].dtype} - {len(y[TARGET_COLUMNS[0]].unique())} unique values")
    print(f"🎯 Target 2 ({TARGET_COLUMNS[1]}): {y[TARGET_COLUMNS[1]].dtype} - {len(y[TARGET_COLUMNS[1]].unique())} unique values")

    return X, y, label_encoders

# Model training and evaluation
def train_random_forest(X, y, optimize_hyperparameters=True):
    print("\n" + "="*50)
    print("🌲 RANDOM FOREST TRAINING (MULTI-OUTPUT)")
    print("="*50)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print(f"📊 Training set: {X_train.shape}")
    print(f"📊 Test set: {X_test.shape}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train separate models for each target
    models = {}
    predictions = {}
    accuracies = {}

    for target_col in TARGET_COLUMNS:
        print(f"\n🎯 Training model for {target_col}...")

        # Determine if this is classification or regression
        is_classification = y[target_col].dtype == 'object' or len(y[target_col].unique()) < 20

        if optimize_hyperparameters and is_classification:
            print(f"🔍 Performing hyperparameter optimization for {target_col}...")

            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }

            rf = RandomForestClassifier(random_state=RANDOM_STATE)
            grid_search = GridSearchCV(
                rf, param_grid, cv=CV_FOLDS, scoring='accuracy', n_jobs=-1
            )
            grid_search.fit(X_train_scaled, y_train[target_col])

            best_model = grid_search.best_estimator_
            print(f"✅ Best parameters for {target_col}: {grid_search.best_params_}")
            print(f"✅ Best CV score for {target_col}: {grid_search.best_score_:.4f}")
        else:
            if is_classification:
                print(f"🌲 Training {target_col} with default parameters (classification)...")
                best_model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                )
            else:
                print(f"🌲 Training {target_col} with default parameters (regression)...")
                best_model = RandomForestRegressor(
                    n_estimators=100,
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                )
            best_model.fit(X_train_scaled, y_train[target_col])

        models[target_col] = best_model

        # Make predictions
        y_pred = best_model.predict(X_test_scaled)
        predictions[target_col] = y_pred

        # Evaluate the model
        if is_classification:
            accuracy = accuracy_score(y_test[target_col], y_pred)
            accuracies[target_col] = accuracy
            print(f"📊 {target_col} Accuracy: {accuracy:.4f}")

            print(f"\n📋 Classification Report for {target_col}:")
            print(classification_report(y_test[target_col], y_pred))

            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_test[target_col], y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {target_col}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.show()
        else:
            mse = mean_squared_error(y_test[target_col], y_pred)
            r2 = r2_score(y_test[target_col], y_pred)
            accuracies[target_col] = r2
            print(f"📊 {target_col} R² Score: {r2:.4f}")
            print(f"📊 {target_col} MSE: {mse:.4f}")

            # Plot actual vs predicted
            plt.figure(figsize=(8, 6))
            plt.scatter(y_test[target_col], y_pred, alpha=0.7)
            plt.plot([y_test[target_col].min(), y_test[target_col].max()],
                    [y_test[target_col].min(), y_test[target_col].max()], 'r--', lw=2)
            plt.xlabel(f'Actual {target_col}')
            plt.ylabel(f'Predicted {target_col}')
            plt.title(f'Actual vs Predicted - {target_col}')
            plt.show()

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\n🔝 Top 10 Most Important Features for {target_col}:")
        print(feature_importance.head(10))

        # Plot feature importance
        plt.figure(figsize=(10, 6))
        top_features = feature_importance.head(15)
        plt.barh(top_features['feature'], top_features['importance'])
        plt.xlabel('Importance')
        plt.title(f'Top 15 Feature Importances - {target_col}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

    # Combined predictions analysis
    print(f"\n🔗 COMBINED PREDICTIONS ANALYSIS")
    print("="*40)

    # Create a combined predictions DataFrame
    results_df = pd.DataFrame({
        f'actual_{TARGET_COLUMNS[0]}': y_test[TARGET_COLUMNS[0]].values,
        f'predicted_{TARGET_COLUMNS[0]}': predictions[TARGET_COLUMNS[0]],
        f'actual_{TARGET_COLUMNS[1]}': y_test[TARGET_COLUMNS[1]].values,
        f'predicted_{TARGET_COLUMNS[1]}': predictions[TARGET_COLUMNS[1]]
    })

    # Calculate combined accuracy (both predictions correct)
    correct_both = (
        (results_df[f'actual_{TARGET_COLUMNS[0]}'] == results_df[f'predicted_{TARGET_COLUMNS[0]}']) &
        (results_df[f'actual_{TARGET_COLUMNS[1]}'] == results_df[f'predicted_{TARGET_COLUMNS[1]}'])
    )
    combined_accuracy = correct_both.mean()
    print(f"🎯 Combined Accuracy (both correct): {combined_accuracy:.4f}")

    print("\n📊 Sample Predictions:")
    print(results_df.head(10))

    return models, scaler, accuracies, results_df

# Prediction function
def make_predictions(models, scaler, X_new):
    """Make predictions on new data for both targets"""
    X_new_scaled = scaler.transform(X_new)

    results = {}
    for target_col in TARGET_COLUMNS:
        model = models[target_col]
        predictions = model.predict(X_new_scaled)

        # Get probabilities if classification
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_new_scaled)
            results[target_col] = {
                'predictions': predictions,
                'probabilities': probabilities
            }
        else:  # Regression
            results[target_col] = {
                'predictions': predictions,
                'probabilities': None
            }

    return results

# Helper function to convert single config to two columns
def split_best_config(df, config_column='best_config', separator='-'):
    """Convert 'size1-size2' format to separate columns"""
    if config_column in df.columns:
        print(f"🔄 Splitting '{config_column}' into {TARGET_COLUMNS}...")
        df[TARGET_COLUMNS] = df[config_column].str.split(separator, expand=True)
        df = df.drop(config_column, axis=1)
        print(f"✅ Split complete! Created columns: {TARGET_COLUMNS}")
    return df

# Main execution
if __name__ == "__main__":
    try:
        # Explore the data
        explore_data(df)

        # Auto-split best_config if it exists and target columns don't
        if 'best_config' in df.columns and not all(col in df.columns for col in TARGET_COLUMNS):
            df = split_best_config(df)

        # Preprocessing options
        ENABLE_PREPROCESSING = True  # Set to False to disable preprocessing
        COLUMNS_TO_REMOVE = []  # Add column names to remove, e.g., ['id', 'timestamp']
        OPTIMIZE_HYPERPARAMETERS = True  # Set to False for faster training

        # Preprocess the data
        df_processed = preprocess_data(
            df,
            remove_columns=COLUMNS_TO_REMOVE,
            enable_preprocessing=ENABLE_PREPROCESSING
        )

        # Prepare features
        X, y, encoders = prepare_features(df_processed)

        # Train the models
        models, scaler, accuracies, results = train_random_forest(
            X, y,
            optimize_hyperparameters=OPTIMIZE_HYPERPARAMETERS
        )

        print("\n" + "="*50)
        print("🎉 MODEL TRAINING COMPLETE!")
        print("="*50)
        print(f"Model Accuracies:")
        for target, acc in accuracies.items():
            print(f"  {target}: {acc:.4f}")

        print("\n💡 To make predictions on new data:")
        print("results = make_predictions(models, scaler, X_new)")
        print("size1_pred = results['size1']['predictions']")
        print("size2_pred = results['size2']['predictions']")

        # Example of making a prediction (uncomment and modify as needed)
        # sample_data = X.iloc[:1]  # Use first row as example
        # pred_results = make_predictions(models, scaler, sample_data)
        # print(f"Sample predictions:")
        # print(f"  Size1: {pred_results['size1']['predictions'][0]}")
        # print(f"  Size2: {pred_results['size2']['predictions'][0]}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Check if 'size1' and 'size2' columns exist in your CSV")
        print("2. Ensure CSV file path is correct")
        print("3. Verify data format and encoding")
        print("4. If using old 'best_config' format, it should auto-convert")
