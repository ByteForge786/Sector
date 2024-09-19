def load_and_preprocess_data(file_path):
    print("Loading data from", file_path)
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows of data")
    
    print("Setting 'label' to null for 'Diversified' parent_sesc and 'Others' label")
    df.loc[(df['parent_sesc'] == 'Diversified') & (df['label'] == 'Others'), 'label'] = np.nan
    
    print("Separating rows with null labels")
    null_label_data = df[df['label'].isnull()]
    df = df.dropna(subset=['label'])
    
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
    
    # Add rows with null labels to test_data
    test_data = pd.concat([test_data, null_label_data])
    
    print(f"Train set size: {len(train_data)}")
    print(f"Test set size: {len(test_data)}")
    
    # Save train and test data to CSV
    train_data.to_csv('train_data.csv', index=False)
    test_data.to_csv('test_data.csv', index=False)
    print("Saved train_data.csv and test_data.csv")
    
    return train_data, test_data
