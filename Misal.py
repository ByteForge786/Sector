import pandas as pd
from sklearn.preprocessing import LabelEncoder
from setfit import SetFitModel, SetFitTrainer
from sklearn.metrics import classification_report
from datasets import Dataset

def load_and_preprocess_data(file_path, feature_columns, eval_percentage=0.05):
    print("Loading data from", file_path)
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows of data")
    
    # Combine feature columns into a single 'text' column
    df['text'] = df.apply(lambda row: '. '.join([f"{col.replace('_', ' ').capitalize()}: {str(row[col])}" 
                                                 for col in feature_columns 
                                                 if pd.notna(row[col]) and str(row[col]).strip() != '']), axis=1)
    
    # Keep only 'text' and 'label' columns
    df = df[['text', 'label']]
    
    # Remove rows with empty text
    df = df[df['text'].notna() & (df['text'].str.strip() != '')]
    
    if df.empty:
        print("No valid data after preprocessing. Please check your input file and feature columns.")
        return None, None
    
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

    # Create evaluation dataset by taking 5% of data from each label
    print("Creating evaluation dataset")
    eval_data = pd.DataFrame()
    for label in balanced_data['label'].unique():
        label_data = balanced_data[balanced_data['label'] == label]
        eval_samples = label_data.sample(frac=eval_percentage, random_state=42)
        eval_data = pd.concat([eval_data, eval_samples])
    
    # Remove evaluation samples from the training data
    train_data = balanced_data.drop(eval_data.index)

    print(f"Training set size: {len(train_data)}")
    print(f"Evaluation set size: {len(eval_data)}")
    
    return train_data, eval_data

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

def train_setfit_model(train_data, eval_data, num_epochs=2, output_dir='best_model'):
    print("Training SetFit model")

    try:
        # Convert DataFrame to Hugging Face Dataset format
        train_dataset = Dataset.from_pandas(train_data)
        eval_dataset = Dataset.from_pandas(eval_data)

        # Rename columns
        train_dataset = train_dataset.rename_column('encoded_label', 'label')
        eval_dataset = eval_dataset.rename_column('encoded_label', 'label')

        # Initialize SetFit model with classification head
        model = SetFitModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", 
                                            use_differentiable_head=True, 
                                            head_params={"n_classes": len(train_data['encoded_label'].unique())})

        trainer = SetFitTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            column_mapping={"text": "text", "label": "label"},
            metric="accuracy",
            num_epochs=num_epochs,  # Set number of epochs
            save_best_model=True,   # Save the best model
            output_dir=output_dir,  # Directory to save the best model
        )

        print("Starting training process...")
        trainer.train()
        eval_metrics = trainer.evaluate()
        print(f"Evaluation metrics: {eval_metrics}")
        print(f"Model training complete. Best model saved to {output_dir}")
    except Exception as e:
        print("An error occurred during model training:", e)
    
    return trainer.model

def predict_unseen_data(model, id_to_label, unseen_data):
    print("Predicting unseen data")
    try:
        # Ensure unseen_data contains 'text' column
        if 'text' not in unseen_data.columns:
            raise ValueError("The 'text' column is missing in unseen_data.")

        # Convert unseen_data to list of texts
        texts = unseen_data['text'].tolist()

        # Make predictions using the fine-tuned model
        predictions = model.predict(texts)
        
        # Convert numeric predictions back to original labels
        predicted_labels = [id_to_label[pred] for pred in predictions]

        # Get prediction probabilities
        prediction_probs = model.predict_proba(texts)
        
        # Add predictions and probabilities to unseen data
        unseen_data['predicted_label'] = predicted_labels
        unseen_data['prediction_probability'] = prediction_probs.max(axis=1)  # Get the max probability for each prediction

        print("Predictions complete")
    except Exception as e:
        print("An error occurred during predictions on unseen data:", e)
    return unseen_data

def main(file_path, feature_columns):
    # Load and preprocess data
    train_data, eval_data = load_and_preprocess_data(file_path, feature_columns)
    
    if train_data is None or train_data.empty:
        print("No data available after preprocessing. Please check your input file and feature columns.")
        return
    
    # Encode labels and get mappings
    train_data, le, label_to_id, id_to_label = encode_labels(train_data)
    eval_data, _, _, _ = encode_labels(eval_data)
    
    # Train SetFit model and save the best one
    model = train_setfit_model(train_data, eval_data)
    
    # Use the remaining data from the original dataset for testing
    test_data = pd.read_csv(file_path)
    test_data = test_data[~test_data.index.isin(train_data.index) & ~test_data.index.isin(eval_data.index)]
    test_data, _, _, _ = encode_labels(test_data)  # Re-encode test data labels
    
    print(f"Test set size: {len(test_data)}")
    
    # Make predictions on test data
    print("Making predictions on test set")
    test_data = predict_unseen_data(model, id_to_label, test_data)
    
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
    unseen_data['text'] = unseen_data.apply(lambda row: '. '.join([f"{col.replace('_', ' ').capitalize()}: {str(row[col])}" 
                                                                   for col in feature_columns 
                                                                   if pd.notna(row[col]) and str(row[col]).strip() != '']), axis=1)
    
    # Remove rows with empty text
    unseen_data = unseen_data[unseen_data['text'].notna() & (unseen_data['text'].str.strip() != '')]
    
    if not unseen_data.empty:
        predicted_unseen_data = predict_unseen_data(model, id_to_label, unseen_data)
        
        # Save unseen data predictions to CSV
        unseen_output_file = 'unseen_data_results.csv'
        print(f"Saving unseen data predictions to {unseen_output_file}")
        predicted_unseen_data.to_csv(unseen_output_file, index=False)
        print("Unseen data predictions saved!")
    else:
        print("No valid unseen data available for prediction.")

if __name__ == "__main__":
    file_path = "your_data.csv"  # Replace with your actual file path
    feature_columns = ['company_name', 'attribute1', 'attribute2']  # Add all your input columns here
    main(file_path, feature_columns)  # Start the main process
