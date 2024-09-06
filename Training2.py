import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
import time
import os
import pickle

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
    
    # Save train and test data to CSV
    train_data.to_csv('train_data.csv', index=False)
    test_data.to_csv('test_data.csv', index=False)
    print("Saved train_data.csv and test_data.csv")
    
    return train_data, test_data

def encode_labels(train_data, test_data):
    print("Encoding labels")
    le = LabelEncoder()
    train_data['encoded_label'] = le.fit_transform(train_data['label'])
    test_data['encoded_label'] = le.transform(test_data['label'])
    print(f"Number of unique labels: {len(le.classes_)}")
    return train_data, test_data, le

def create_features(data, feature_columns, cache_dir='embedding_cache', prefix='train'):
    print(f"Creating features for {prefix} data using SentenceTransformer")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def combine_features(row):
        return ', '.join(f"{col}: {row[col]}" for col in feature_columns)
    
    cache_path = os.path.join(cache_dir, f"{prefix}_embeddings.pkl")
    os.makedirs(cache_dir, exist_ok=True)
    
    if os.path.exists(cache_path):
        print("Loading embeddings from cache")
        with open(cache_path, 'rb') as f:
            embeddings = pickle.load(f)
    else:
        print("Combining features")
        combined = data.apply(combine_features, axis=1)
        
        print("Encoding features")
        embeddings = model.encode(combined.tolist(), show_progress_bar=True)
        
        print("Caching embeddings")
        with open(cache_path, 'wb') as f:
            pickle.dump(embeddings, f)
    
    return embeddings

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

def save_model(model, le, model_dir='saved_model'):
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'best_model.pkl')
    le_path = os.path.join(model_dir, 'label_encoder.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(le_path, 'wb') as f:
        pickle.dump(le, f)
    print(f"Best model saved to {model_path}")
    print(f"Label encoder saved to {le_path}")

def predict_single_row(row, feature_columns, model, le):
    features = ', '.join(f"{col}: {row[col]}" for col in feature_columns)
    embedding = SentenceTransformer('all-MiniLM-L6-v2').encode([features])
    prediction = model.predict(embedding)
    probabilities = model.predict_proba(embedding)[0]
    predicted_label = le.inverse_transform(prediction)[0]
    return predicted_label, probabilities

def predict_test_data(test_data, feature_columns, model, le):
    print("Making predictions on test set")
    results = []
    for _, row in test_data.iterrows():
        predicted_label, probabilities = predict_single_row(row, feature_columns, model, le)
        result = row.to_dict()
        result['predicted_label'] = predicted_label
        for i, class_name in enumerate(le.classes_):
            result[f'probability_{class_name}'] = probabilities[i]
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    # Ensure the order of columns matches the original test_data
    original_columns = test_data.columns.tolist()
    new_columns = [col for col in results_df.columns if col not in original_columns]
    final_column_order = original_columns + new_columns
    
    return results_df[final_column_order]

def main(file_path, feature_columns):
    # Load and preprocess data
    train_data, test_data = load_and_preprocess_data(file_path)
    
    # Encode labels
    train_data, test_data, le = encode_labels(train_data, test_data)
    
    # Create features for training data
    train_embeddings = create_features(train_data, feature_columns, prefix='train')
    
    # Train and tune XGBoost model
    model = train_and_tune_xgboost(train_embeddings, train_data['encoded_label'])
    
    # Save the best model and label encoder
    save_model(model, le)
    
    # Predict on test data
    test_results = predict_test_data(test_data, feature_columns, model, le)
    
    # Calculate and print classification report
    print("\nClassification Report:")
    report = classification_report(test_results['label'], test_results['predicted_label'])
    print(report)
    
    # Save results to CSV
    output_file = 'test_results.csv'
    print(f"Saving results to {output_file}")
    test_results.to_csv(output_file, index=False)
    print("Done!")

if __name__ == "__main__":
    file_path = "your_data.csv"  # Replace with your actual file path
    feature_columns = ['company_name', 'attribute1', 'attribute2']  # Add all your input columns here
    main(file_path, feature_columns)
