import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

def calculate_per_class_accuracy(y_true, y_pred):
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate per-class accuracy
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    return per_class_accuracy

def main(predictions_file):
    # Load the predictions
    df = pd.read_csv(predictions_file)
    
    # Ensure 'label' and 'predicted_label' columns exist
    if 'label' not in df.columns or 'predicted_label' not in df.columns:
        raise ValueError("The CSV file must contain 'label' and 'predicted_label' columns")
    
    # Filter rows where both label and predicted_label are present
    valid_data = df.dropna(subset=['label', 'predicted_label'])
    
    if not valid_data.empty:
        print("Calculating accuracy for data with both original and predicted labels:")
        # Calculate per-class accuracy
        per_class_acc = calculate_per_class_accuracy(valid_data['label'], valid_data['predicted_label'])
        
        # Get unique labels
        unique_labels = np.unique(valid_data['label'])
        
        # Print per-class accuracy
        print("\nPer-class Accuracy:")
        for label, accuracy in zip(unique_labels, per_class_acc):
            print(f"Class {label}: {accuracy:.4f}")
        
        # Calculate and print overall accuracy
        overall_accuracy = (valid_data['label'] == valid_data['predicted_label']).mean()
        print(f"\nOverall Accuracy: {overall_accuracy:.4f}")
        
        # Print label distribution
        print("\nLabel distribution:")
        print(valid_data['label'].value_counts(normalize=True))
    else:
        print("No data found with both original and predicted labels present.")

if __name__ == "__main__":
    predictions_file = "predictions_output.csv"  # Replace with your predictions file path
    main(predictions_file)
