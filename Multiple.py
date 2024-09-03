import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    label_counts = df['label'].value_counts()
    min_count = label_counts.min()
    
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()
    
    for label, count in label_counts.items():
        label_data = df[df['label'] == label]
        if count > min_count:
            train_data = pd.concat([train_data, label_data.sample(n=min_count)])
            test_data = pd.concat([test_data, label_data[~label_data.index.isin(train_data.index)]])
        else:
            train_data = pd.concat([train_data, label_data])
    
    return train_data, test_data

def encode_labels(train_data, test_data):
    le = LabelEncoder()
    train_data['encoded_label'] = le.fit_transform(train_data['label'])
    test_data['encoded_label'] = le.transform(test_data['label'])
    return train_data, test_data, le

def create_features(train_data, test_data, feature_columns):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def combine_features(row):
        return ' '.join(str(row[col]) for col in feature_columns)
    
    train_combined = train_data.apply(combine_features, axis=1)
    test_combined = test_data.apply(combine_features, axis=1)
    
    train_embeddings = model.encode(train_combined.tolist())
    test_embeddings = model.encode(test_combined.tolist())
    return train_embeddings, test_embeddings

def train_and_tune_xgboost(X_train, y_train):
    param_grid = {
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    random_search = RandomizedSearchCV(xgb, param_distributions=param_grid, n_iter=20, cv=3, random_state=42)
    random_search.fit(X_train, y_train)
    
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
    predictions = model.predict(test_embeddings)
    predicted_labels = le.inverse_transform(predictions)
    
    # Add predictions to test data
    test_data['predicted_label'] = predicted_labels
    
    # Calculate and print classification report
    report = classification_report(test_data['label'], test_data['predicted_label'])
    print(report)
    
    # Save results to CSV
    test_data.to_csv('test_results.csv', index=False)

if __name__ == "__main__":
    file_path = "your_data.csv"  # Replace with your actual file path
    feature_columns = ['company_name', 'attribute1', 'attribute2']  # Add all your input columns here
    main(file_path, feature_columns)
