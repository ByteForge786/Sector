import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from tqdm import tqdm
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil
import json

def load_model_and_encoder(model_dir='saved_model'):
    model_path = os.path.join(model_dir, 'best_model.pkl')
    le_path = os.path.join(model_dir, 'label_encoder.pkl')
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(le_path, 'rb') as f:
        le = pickle.load(f)
    
    return model, le

def create_embeddings_batch(args):
    batch, feature_columns, model_name = args
    model = SentenceTransformer(model_name)
    
    def combine_features(row):
        return ', '.join(f"{col}: {row[col]}" for col in feature_columns)
    
    combined_features = [combine_features(row) for _, row in batch.iterrows()]
    return model.encode(combined_features, show_progress_bar=False)

def create_embeddings(df, feature_columns, batch_size=1000, num_processes=None):
    model_name = 'paraphrase-MiniLM-L3-v2'  # Smaller, faster model
    
    if num_processes is None:
        num_processes = max(1, psutil.cpu_count(logical=False) - 1)  # Use physical cores minus 1
    
    if len(df) <= batch_size:
        return create_embeddings_batch((df, feature_columns, model_name))
    
    batches = [df[i:i+batch_size] for i in range(0, len(df), batch_size)]
    
    all_embeddings = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(create_embeddings_batch, (batch, feature_columns, model_name)) for batch in batches]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Creating embeddings"):
            all_embeddings.append(future.result())
    
    return np.vstack(all_embeddings)

def batch_predict(df, feature_columns, model, le, batch_size=10000):
    embeddings = create_embeddings(df, feature_columns, batch_size)
    
    all_predictions = []
    all_max_probabilities = []
    
    for i in tqdm(range(0, len(df), batch_size), desc="Making predictions"):
        batch_embeddings = embeddings[i:i+batch_size]
        predictions = model.predict(batch_embeddings)
        probabilities = model.predict_proba(batch_embeddings)
        all_predictions.extend(predictions)
        all_max_probabilities.extend(np.max(probabilities, axis=1))
    
    predicted_labels = le.inverse_transform(all_predictions)
    
    df['predicted_label'] = predicted_labels
    df['max_probability'] = all_max_probabilities
    
    return df

def df_to_json(df, output_file):
    json_data = df.to_json(orient='records')
    with open(output_file, 'w') as f:
        json.dump(json.loads(json_data), f, indent=2)
    print(f"Saved JSON data to {output_file}")

def main(input_file, output_file, feature_columns):
    print("Loading data...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows of data")
    
    print("Loading model and label encoder...")
    model, le = load_model_and_encoder()
    
    print("Starting prediction...")
    result_df = batch_predict(df, feature_columns, model, le)
    
    print(f"Saving results to CSV: {output_file}")
    result_df.to_csv(output_file, index=False)
    
    json_output_file = output_file.rsplit('.', 1)[0] + '.json'
    print(f"Saving results to JSON: {json_output_file}")
    df_to_json(result_df, json_output_file)
    
    print("Done!")

if __name__ == "__main__":
    input_file = "your_test_data.csv"  # Replace with your input file path
    output_file = "predictions_output.csv"  # Replace with your desired output file path
    feature_columns = ['company_name', 'attribute1', 'attribute2']  # Add all your input columns here
    main(input_file, output_file, feature_columns)
