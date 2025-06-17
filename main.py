import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score, 
                             classification_report, confusion_matrix, roc_curve, precision_recall_curve)
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Input
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from sklearn.model_selection import GridSearchCV
import lime
import lime.lime_tabular

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(dataset_path):
    """Load dataset from the given path."""
    try:
        df = pd.read_csv(dataset_path)
        logging.info("Dataset loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise

def preprocess_data(df):
    """Preprocess the dataset."""
    try:
        # Assuming 'Churn' is the target column
        y = df['Churn']
        X = df.drop(columns=['Churn'])

        # Convert non-numeric columns to numeric, coercing errors to NaN
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')

        # Fill NaNs with column means
        X.fillna(X.mean(), inplace=True)

        # Check if 'MonthlyCharges' and 'Tenure' are included
        if 'MonthlyCharges' not in X.columns:
            X['MonthlyCharges'] = np.random.uniform(20, 100, size=len(X))
            logging.info("Filled 'MonthlyCharges' with random values.")
        if 'Tenure' not in X.columns:
            X['Tenure'] = np.random.randint(1, 73, size=len(X))
            logging.info("Filled 'Tenure' with random values.")

        logging.info("Data preprocessing completed.")
        return X, y
    except Exception as e:
        logging.error(f"Error in data preprocessing: {e}")
        raise

def apply_smote(X, y):
    """Apply SMOTE to handle class imbalance."""
    try:
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        logging.info("SMOTE applied successfully.")
        return X_resampled, y_resampled
    except Exception as e:
        logging.error(f"Error applying SMOTE: {e}")
        raise

def create_lstm_model(input_shape):
    """Create and compile an LSTM model."""
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train and evaluate models."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),  # Increased max_iter
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(eval_metric='mlogloss', use_label_encoder=False)
    }

    results = {}
    y_pred_probs = {}
    fitted_models = {}

    for name, model in models.items():
        pipeline = Pipeline(steps=[('classifier', model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_pred_prob = pipeline.predict_proba(X_test)[:, 1]
        y_pred_probs[name] = y_pred_prob
        results[name] = {
            "accuracy": pipeline.score(X_test, y_test),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred_prob)
        }
        fitted_models[name] = pipeline
        logging.info(f"\nModel: {name}")
        logging.info(f"{classification_report(y_test, y_pred)}")
        logging.info(f"Recall: {results[name]['recall']:.2f}")
        logging.info(f"F1 Score: {results[name]['f1']:.2f}")

    return results, y_pred_probs, fitted_models

def tune_random_forest(X_train, y_train):
    """Tune Random Forest hyperparameters."""
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [4, 6, 8, 10],
        'criterion': ['gini', 'entropy']
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='roc_auc')
    grid_search.fit(X_train, y_train)
    logging.info(f"Best parameters for Random Forest: {grid_search.best_params_}")
    return grid_search.best_estimator_

def train_lstm_model(X_train, X_test, y_train, y_test):
    """Train and evaluate the LSTM model."""
    max_len = X_train.shape[1]

    # Convert to numeric and handle non-numeric values
    X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
    X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)

    X_train_pad = pad_sequences(X_train.values, maxlen=max_len)
    X_test_pad = pad_sequences(X_test.values, maxlen=max_len)

    lstm_model = create_lstm_model((max_len, 1))
    lstm_model.fit(X_train_pad, y_train, epochs=10, batch_size=32, validation_split=0.2)

    y_pred_lstm = (lstm_model.predict(X_test_pad) > 0.5).astype(int).flatten()
    y_pred_lstm_prob = lstm_model.predict(X_test_pad).flatten()

    results = {
        "accuracy": (y_pred_lstm == y_test).mean(),
        "precision": precision_score(y_test, y_pred_lstm),
        "recall": recall_score(y_test, y_pred_lstm),
        "f1": f1_score(y_test, y_pred_lstm),
        "roc_auc": roc_auc_score(y_test, y_pred_lstm_prob)
    }

    logging.info(f"\nModel: LSTM")
    logging.info(f"{classification_report(y_test, y_pred_lstm)}")
    logging.info(f"Recall: {results['recall']:.2f}")
    logging.info(f"F1 Score: {results['f1']:.2f}")

    return results, y_pred_lstm_prob

def visualize_roc_curves(y_test, y_pred_probs, results):
    """Visualize ROC Curves for model comparison."""
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")

    for name, y_pred_prob in y_pred_probs.items():
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {results[name]["roc_auc"]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curve for Model Comparison', fontsize=16)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True)
    plt.show()

def visualize_precision_recall_curves(y_test, y_pred_probs, results):
    """Visualize Precision-Recall Curves for model comparison."""
    plt.figure(figsize=(10, 6))

    for name, y_pred_prob in y_pred_probs.items():
        precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
        plt.plot(recall, precision, label=f'{name} (AUC = {results[name]["roc_auc"]:.2f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curve', fontsize=16)
    plt.legend(loc='lower left', fontsize=12)
    plt.grid(True)
    plt.show()

def visualize_confusion_matrices(models, X_test, y_test):
    """Visualize Confusion Matrices for each model."""
    for name, model in models.items():
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
        plt.title(f'Confusion Matrix for {name}', fontsize=14)
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        plt.show()

def retention_strategy(row):
    if row['Churn_Risk'] > 0.9:
        if 'MonthlyCharges' in row and row['MonthlyCharges'] > 80:
            return 'Offer premium support'
        else:
            return 'Enhance customer service'
    elif row['Churn_Risk'] > 0.7:
        if 'Tenure' in row and row['Tenure'] < 12:
            return 'Offer discounts for new customers'
        else:
            return 'Offer discounts'
    elif row['Churn_Risk'] > 0.5:
        if 'MonthlyCharges' in row and row['MonthlyCharges'] < 30:
            return 'Provide loyalty rewards for low spenders'
        else:
            return 'Provide loyalty rewards'
    else:
        return 'Standard follow-up'

def develop_retention_strategy(df, X_test, y_test, pipeline):
    """Develop retention strategy based on churn risk."""
    X_test_df = pd.DataFrame(X_test, columns=df.columns.drop('Churn'))
    X_test_df['Churn_Risk'] = pipeline.predict_proba(X_test)[:, 1]
    high_risk_customers = X_test_df[X_test_df['Churn_Risk'] > 0.5]

    # Ensure 'MonthlyCharges' and 'Tenure' are in the DataFrame
    if 'MonthlyCharges' not in high_risk_customers.columns:
        high_risk_customers['MonthlyCharges'] = np.random.uniform(20, 100, size=len(high_risk_customers))
    if 'Tenure' not in high_risk_customers.columns:
        high_risk_customers['Tenure'] = np.random.randint(1, 73, size=len(high_risk_customers))

    # Apply the retention strategy
    high_risk_customers.loc[:, 'Retention_Strategy'] = high_risk_customers.apply(retention_strategy, axis=1)

    # Print the results
    logging.info(high_risk_customers[['Churn_Risk', 'Retention_Strategy', 'MonthlyCharges', 'Tenure']])
    return high_risk_customers

def export_results(high_risk_customers, results, export_path):
    """Export high-risk customer data and model performance results."""
    try:
        os.makedirs(export_path, exist_ok=True)
        high_risk_customers.to_csv(f'{export_path}high_risk_customers.csv', index=False)
        performance_df = pd.DataFrame(results).T
        performance_df.to_csv(f'{export_path}model_performance_results.csv', index=True)
        logging.info("Model performance results exported as 'model_performance_results.csv'.")
        logging.info("High-risk customer data exported as 'high_risk_customers.csv'.")
    except Exception as e:
        logging.error(f"Error exporting results: {e}")
        raise

def churn_prediction_automation(dataset_path, export_path):
    """Main function to run the churn prediction automation."""
    df = load_data(dataset_path)
    X, y = preprocess_data(df)
    X_resampled, y_resampled = apply_smote(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

    results, y_pred_probs, fitted_models = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    lstm_results, y_pred_lstm_prob = train_lstm_model(X_train, X_test, y_train, y_test)
    results["LSTM"] = lstm_results
    y_pred_probs["LSTM"] = y_pred_lstm_prob

    # Forcefully select Random Forest as the best model
    best_model_name = "Random Forest"
    logging.info(f"Best model (forcefully selected): {best_model_name}")

    # Perform hyperparameter tuning for Random Forest
    tuned_rf = tune_random_forest(X_train, y_train)
    fitted_models["Random Forest"] = Pipeline(steps=[('classifier', tuned_rf)])
    fitted_models["Random Forest"].fit(X_train, y_train)
    y_pred = fitted_models["Random Forest"].predict(X_test)
    y_pred_prob = fitted_models["Random Forest"].predict_proba(X_test)[:, 1]
    results["Random Forest"] = {
        "accuracy": fitted_models["Random Forest"].score(X_test, y_test),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_prob)
    }
    y_pred_probs["Random Forest"] = y_pred_prob

    visualize_roc_curves(y_test, y_pred_probs, results)
    visualize_precision_recall_curves(y_test, y_pred_probs, results)
    visualize_confusion_matrices(fitted_models, X_test, y_test)

    # Use the RandomForest pipeline for retention strategy
    pipeline = fitted_models["Random Forest"]

    high_risk_customers = develop_retention_strategy(df, X_test, y_test, pipeline)
    export_results(high_risk_customers, results, export_path)

    # LIME explanation
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train.values, 
        feature_names=X_train.columns, 
        class_names=['Not Churn', 'Churn'], 
        discretize_continuous=True
    )

    # Select an instance to explain
    instance_to_explain = X_test.iloc[0]

    # Explain the instance
    exp = explainer.explain_instance(
        instance_to_explain.values, 
        pipeline.predict_proba, 
        num_features=10
    )

    # Print the explanation
    exp.show_in_notebook(show_table=True)
    exp.save_to_file('lime_explanation.html')

    # Log the explanation
    logging.info(exp.as_list())

# Run the automation function
if __name__ == "__main__":
    churn_prediction_automation(dataset_path='/Users/jeeveshsingale/PycharmProjects/pythonProject1/customer_churn.csv', export_path='output/')