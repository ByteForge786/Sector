import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from setfit import SetFitModel, SetFitTrainer
from sklearn.metrics import classification_report

def load_and_preprocess_data(file_path, feature_columns):
    print("Loading data from", file_path)
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows of data")
    
    # Combine feature columns into a single 'text' column
    df['text'] = df.apply(lambda row: '. '.join([f"{col.replace('_', ' ').capitalize()}: {row[col]}" for col in feature_columns if pd.notna(row[col]) and row[col] != '']), axis=1)
    
    # Keep only 'text' and 'label' columns
    df = df[['text', 'label']]
    
    print("Counting labels and finding minimum count")
    label_counts = df['label'].value_counts()
    min_count = label_counts.min()
    print(f"Minimum label count: {min_count}")
    
    print("Balancing dataset")
    balanced_data = pd.DataFrame()
    for label, count in label_counts.items():
        label_data = df[df['label'] == label]
        if count > min_count:
            balanced_data = pd.concat([balanced_data, label_data.sample(n=min_count)])
        else:
            balanced_data = pd.concat([balanced_data, label_data])
    
    print(f"Balanced dataset size: {len(balanced_data)}")
    return balanced_data

def encode_labels(data):
    print("Encoding labels")
    le = LabelEncoder()
    
    # Fit the label encoder and transform labels into numeric IDs
    data['encoded_label'] = le.fit_transform(data['label'])
    
    # Create label-to-ID and ID-to-label mappings
    label_to_id = {label: idx for idx, label in enumerate(le.classes_)}
    id_to_label = {idx: label for idx, label in enumerate(le.classes_)}
    
    print(f"Number of unique labels: {len(le.classes_)}")
    return data, le, label_to_id, id_to_label

def train_setfit_model(train_data, eval_data, num_iterations=20, batch_size=16):
    print(f"Training SetFit model for {num_iterations} iterations")
    model = SetFitModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        column_mapping={"text": "text", "label": "encoded_label"},
        num_iterations=num_iterations,
        batch_size=batch_size,
        metric="accuracy",
    )

    trainer.train()
    eval_metrics = trainer.evaluate()
    print(f"Evaluation metrics: {eval_metrics}")
    
    print("Model training complete")
    return trainer.model

def predict_unseen_data(model, id_to_label, unseen_data):
    print("Predicting unseen data")
    
    # Make predictions using the fine-tuned model
    predictions = model.predict(unseen_data['text'].tolist())
    
    # Convert numeric predictions back to original labels
    predicted_labels = [id_to_label[pred] for pred in predictions]
    
    # Add predictions to unseen data
    unseen_data['predicted_label'] = predicted_labels
    
    print("Predictions complete")
    return unseen_data

def main(file_path, feature_columns, num_iterations=20, batch_size=16, test_size=0.2, eval_size=0.1):
    # Load and preprocess data
    data = load_and_preprocess_data(file_path, feature_columns)
    
    # Encode labels and get mappings
    data, le, label_to_id, id_to_label = encode_labels(data)
    
    # Split data into train, eval, and test sets
    train_eval_data, test_data = train_test_split(data, test_size=test_size, stratify=data['label'], random_state=42)
    train_data, eval_data = train_test_split(train_eval_data, test_size=eval_size, stratify=train_eval_data['label'], random_state=42)
    
    print(f"Train set size: {len(train_data)}")
    print(f"Evaluation set size: {len(eval_data)}")
    print(f"Test set size: {len(test_data)}")
    
    # Train SetFit model
    model = train_setfit_model(train_data, eval_data, num_iterations, batch_size)
    
    # Make predictions on test data
    print("Making predictions on test set")
    predictions = model.predict(test_data['text'].tolist())
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
    
    # Combine feature columns for unseen data
    unseen_data['text'] = unseen_data.apply(lambda row: '. '.join([f"{col.replace('_', ' ').capitalize()}: {row[col]}" for col in feature_columns if pd.notna(row[col]) and row[col] != '']), axis=1)
    
    predicted_unseen_data = predict_unseen_data(model, id_to_label, unseen_data)
    
    # Save unseen data predictions to CSV
    unseen_output_file = 'unseen_data_results.csv'
    print(f"Saving unseen data predictions to {unseen_output_file}")
    predicted_unseen_data.to_csv(unseen_output_file, index=False)
    print("Unseen data predictions saved!")

if __name__ == "__main__":
    file_path = "your_data.csv"  # Replace with your actual file path
    feature_columns = ['company_name', 'attribute1', 'attribute2']  # Add all your input columns here
    main(file_path, feature_columns, num_iterations=20, batch_size=16)  # You can adjust the number of iterations and batch size here
