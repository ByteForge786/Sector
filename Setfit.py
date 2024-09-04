import pandas as pd
from sklearn.preprocessing import LabelEncoder
from setfit import SetFitModel, SetFitTrainer
from sklearn.metrics import classification_report

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
    
    def combine_features(row):
        return ' '.join(str(row[col]) for col in feature_columns)
    
    print("Combining features for train set")
    train_combined = train_data.apply(combine_features, axis=1)
    print("Combining features for test set")
    test_combined = test_data.apply(combine_features, axis=1)
    
    return train_combined.tolist(), test_combined.tolist()

def train_setfit_model(train_texts, train_labels):
    print("Fine-tuning SetFit model")
    # Load a pre-trained SetFit model
    model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")
    
    # Create a trainer for the SetFit model
    trainer = SetFitTrainer(
        model=model,
        train_dataset=(train_texts, train_labels),
        metric="accuracy"
    )
    
    # Train the model
    trainer.train()
    
    return model

def predict_unseen_data(model, le, unseen_data, feature_columns):
    print("Predicting unseen data")
    
    # Combine the feature columns to create the input text
    def combine_features(row):
        return ' '.join(str(row[col]) for col in feature_columns)
    
    # Process the unseen data
    unseen_texts = unseen_data.apply(combine_features, axis=1).tolist()
    
    # Make predictions using the fine-tuned model
    predictions = model.predict(unseen_texts)
    
    # Convert the encoded labels back to original labels
    predicted_labels = le.inverse_transform(predictions)
    
    # Add predictions to unseen data
    unseen_data['predicted_label'] = predicted_labels
    
    print("Predictions complete")
    
    return unseen_data

def main(file_path, feature_columns):
    # Load and preprocess data
    train_data, test_data = load_and_preprocess_data(file_path)
    
    # Encode labels
    train_data, test_data, le = encode_labels(train_data, test_data)
    
    # Create features
    train_texts, test_texts = create_features(train_data, test_data, feature_columns)
    
    # Train SetFit model
    model = train_setfit_model(train_texts, train_data['encoded_label'])
    
    # Make predictions on test data
    print("Making predictions on test set")
    predictions = model.predict(test_texts)
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
    print("Done!")
    
    # Example prediction on unseen data
    unseen_data_path = "unseen_data.csv"  # Replace with your actual unseen data file path
    unseen_data = pd.read_csv(unseen_data_path)
    predicted_unseen_data = predict_unseen_data(model, le, unseen_data, feature_columns)
    
    # Save unseen data predictions to CSV
    unseen_output_file = 'unseen_data_results.csv'
    print(f"Saving unseen data predictions to {unseen_output_file}")
    predicted_unseen_data.to_csv(unseen_output_file, index=False)
    print("Unseen data predictions saved!")

if __name__ == "__main__":
    file_path = "your_data.csv"  # Replace with your actual file path
    feature_columns = ['company_name', 'attribute1', 'attribute2']  # Add all your input columns here
    main(file_path, feature_columns)
