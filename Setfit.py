
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from setfit import SetFitModel, SetFitTrainer
from sklearn.metrics import classification_report
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
    return train_data, test_data

def encode_labels(train_data, test_data):
    print("Encoding labels")
    le = LabelEncoder()
    
    # Fit the label encoder and transform labels into numeric IDs
    train_data['encoded_label'] = le.fit_transform(train_data['label'])
    test_data['encoded_label'] = le.transform(test_data['label'])
    
    # Create label-to-ID and ID-to-label mappings
    label_to_id = {label: idx for idx, label in enumerate(le.classes_)}
    id_to_label = {idx: label for idx, label in enumerate(le.classes_)}
    
    print(f"Number of unique labels: {len(le.classes_)}")
    return train_data, test_data, le, label_to_id, id_to_label

def create_features(data, feature_columns, cache_prefix=''):
    print(f"Creating features for {cache_prefix} using SentenceTransformer")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def combine_features(row):
        return ' '.join(str(row[col]) for col in feature_columns)
    
    data_combined = data.apply(combine_features, axis=1).tolist()
    embeddings = model.encode(data_combined, show_progress_bar=True)
    
    return embeddings

def split_train_eval_data(train_data, eval_ratio=0.05):
    print(f"Splitting train data into training and evaluation sets with evaluation ratio: {eval_ratio}")
    eval_data = pd.DataFrame()
    
    # For each label, take 5% of data for evaluation
    for label in train_data['label'].unique():
        label_data = train_data[train_data['label'] == label]
        eval_size = max(1, int(len(label_data) * eval_ratio))
        eval_samples = label_data.sample(n=eval_size, random_state=42)
        eval_data = pd.concat([eval_data, eval_samples])
        
        # Remove selected eval samples from training data
        train_data = train_data.drop(eval_samples.index)

    print(f"Final Training set size: {len(train_data)}, Evaluation set size: {len(eval_data)}")
    return train_data, eval_data

def train_setfit_model(train_texts, train_labels, eval_texts, eval_labels, num_epochs=3):
    print(f"Training SetFit model for {num_epochs} epochs")
    model = SetFitModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_texts,
        eval_dataset=eval_texts,
        train_labels=train_labels,
        eval_labels=eval_labels,
        metric="accuracy",
        num_epochs=num_epochs  # Specify the number of epochs
    )

    trainer.train()
    trainer.evaluate()
    
    print("Model training complete")
    return trainer

def predict_unseen_data(model, id_to_label, unseen_data, feature_columns):
    print("Predicting unseen data")
    
    def combine_features(row):
        return ' '.join(str(row[col]) for col in feature_columns)
    
    unseen_texts = unseen_data.apply(combine_features, axis=1).tolist()
    
    # Make predictions using the fine-tuned model
    predictions = model.predict(unseen_texts)
    
    # Convert numeric predictions back to original labels
    predicted_labels = [id_to_label[pred] for pred in predictions]
    
    # Add predictions to unseen data
    unseen_data['predicted_label'] = predicted_labels
    
    print("Predictions complete")
    return unseen_data

def main(file_path, feature_columns, num_epochs=3):
    # Load and preprocess data
    train_data, test_data = load_and_preprocess_data(file_path)
    
    # Encode labels and get mappings
    train_data, test_data, le, label_to_id, id_to_label = encode_labels(train_data, test_data)
    
    # Split training data into training and evaluation sets
    train_data, eval_data = split_train_eval_data(train_data)
    
    # Create features for train, eval, and test data
    train_embeddings = create_features(train_data, feature_columns, cache_prefix='train')
    eval_embeddings = create_features(eval_data, feature_columns, cache_prefix='eval')
    test_embeddings = create_features(test_data, feature_columns, cache_prefix='test')
    
    # Train SetFit model with evaluation data
    model = train_setfit_model(train_embeddings, train_data['encoded_label'], eval_embeddings, eval_data['encoded_label'], num_epochs)
    
    # Make predictions on test data
    print("Making predictions on test set")
    predictions = model.predict(test_embeddings)
    predicted_labels = [id_to_label[pred] for pred in predictions]
    
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
    print("Done!")
    
    # Example prediction on unseen data
    unseen_data_path = "unseen_data.csv"  # Replace with your actual unseen data file path
    unseen_data = pd.read_csv(unseen_data_path)
    predicted_unseen_data = predict_unseen_data(model, id_to_label, unseen_data, feature_columns)
    
    # Save unseen data predictions to CSV
    unseen_output_file = 'unseen_data_results.csv'
    print(f"Saving unseen data predictions to {unseen_output_file}")
    predicted_unseen_data.to_csv(unseen_output_file, index=False)
    print("Unseen data predictions saved!")

if __name__ == "__main__":
    file_path = "your_data.csv"  # Replace with your actual file path
    feature_columns = ['company_name', 'attribute1', 'attribute2']  # Add all your input columns here
    main(file_path, feature_columns, num_epochs=5)  # You can adjust the number of epochs here
