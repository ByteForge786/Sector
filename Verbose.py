import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
import time

def load_and_preprocess_data(file_path):
    print("Loading data from", file_path)
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows of data")
    
    print("Counting labels and finding minimum count")
    label_counts = df['label'].value_counts()
    min_count = label_counts.min()
    print(f"Minimum label count: {min_count}")
    
    print("Splitting data into train and test sets")
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()
    
    for label, count in label_counts.items():
        label_data = df[df['label'] == label]
        if count > min_count:
            train_data = pd.concat([train_data, label_data.sample(n=min_count)])
            test_data = pd.concat([test_data, label_data[~label_data.index.isin(train_data.index)]])
        else:
            train_data = pd.concat([train_data, label_data])
    
    print(f"Train set size: {len(train_data)}")
    print(f"Test set size: {len(test_data)}")
    return train_data, test_data

def encode_labels(train_data, test_data):
    print("Encoding labels")
    le = LabelEncoder()
    train_data['encoded_label'] = le.fit_transform(train_data['label'])
    test_data['encoded_label'] = le.transform(test_data['label'])
    print(f"Number of unique labels: {len(le.classes_)}")
    return train_data, test_data, le

def create_features(train_data, test_data, feature_columns):
    print("Creating features using SentenceTransformer")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def combine_features(row):
        return ' '.join(str(row[col]) for col in feature_columns)
    
    print("Combining features for train set")
    train_combined = train_data.apply(combine_features, axis=1)
    print("Combining features for test set")
    test_combined = test_data.apply(combine_features, axis=1)
    
    print("Encoding train set")
    train_embeddings = model.encode(train_combined.tolist(), show_progress_bar=True)
    print("Encoding test set")
    test_embeddings = model.encode(test_combined.tolist(), show_progress_bar=True)
    return train_embeddings, test_embeddings

def train_and_tune_xgboost(X_train, y_train):
    print("Training and tuning XGBoost model")
    param_grid = {
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    random_search = RandomizedSearchCV(xgb, param_distributions=param_grid, n_iter=20, cv=3, random_state=42, verbose=2, n_jobs=-1)
    
    start_time = time.time()
    random_search.fit(X_train, y_train)
    end_time = time.time()
    
    print(f"Best parameters found: {random_search.best_params_}")
    print(f"Best cross-validation score: {random_search.best_score_:.4f}")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    
    return random_search.best_estimator_

def main(file_path, feature_columns):
    # Load and preprocess data
    train_data, test_data = load_and_preprocess_data(file_path)
    
    # Encode labels
    train_data, test_data, le = encode_labels(train_data, test_data)
    
    # Create features
    train_embeddings, test_embeddings = create_features(train_data, test_data, feature_columns)
    
    # Train and tune XGBoost model
    model = train_and_tune_xgboost(train_embeddings, train_data['encoded_label'])
    
    # Make predictions
    print("Making predictions on test set")
    predictions = model.predict(test_embeddings)
    predicted_labels = le.inverse_transform(predictions)
    
    # Add predictions to test data
    test_data['predicted_label'] = predicted_labels
    
    # Calculate and print classification report
    print("\nClassification Report:")
    report = classification_report(test_data['label'], test_data['predicted_label'])
    print(report)
    
    # Save results to CSV
    output_file = 'test_results.csv'
    print(f"Saving results to {output_file}")
    test_data.to_csv(output_file, index=False)
    print("
