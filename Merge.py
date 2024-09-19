import pandas as pd

def merge_csv_files(file1_path, file2_path, output_path):
    # Read the CSV files
    print(f"Reading {file1_path}...")
    df1 = pd.read_csv(file1_path)
    print(f"Reading {file2_path}...")
    df2 = pd.read_csv(file2_path)

    # Merge the dataframes on 'rdmid'
    print("Merging dataframes...")
    merged_df = pd.merge(df1, df2[['rdmid', 'prediction_1', 'prediction_score_1', 
                                   'prediction_2', 'prediction_score_2', 
                                   'prediction_3', 'prediction_score_3']], 
                         on='rdmid', how='left')

    # Save the merged dataframe to a new CSV file
    print(f"Saving merged data to {output_path}...")
    merged_df.to_csv(output_path, index=False)
    print("Merge complete!")

    # Print some information about the merge
    print(f"\nOriginal file 1 shape: {df1.shape}")
    print(f"Original file 2 shape: {df2.shape}")
    print(f"Merged file shape: {merged_df.shape}")
    
    # Check for any rdmids that didn't get matches
    unmatched = merged_df[merged_df['prediction_1'].isna()]['rdmid']
    if not unmatched.empty:
        print(f"\nWarning: {len(unmatched)} rdmids from file 1 did not find matches in file 2.")
        print("First few unmatched rdmids:", unmatched.head().tolist())

if __name__ == "__main__":
    file1_path = "path_to_your_first_csv.csv"  # CSV with rdmid and other columns
    file2_path = "path_to_your_second_csv.csv"  # CSV with rdmid, predictions, and scores
    output_path = "merged_output.csv"  # Path for the output CSV

    merge_csv_files(file1_path, file2_path, output_path)
