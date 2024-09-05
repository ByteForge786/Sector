 # Install necessary libraries
!pip install datasets setfit pandas

import pandas as pd
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset

# Load your own dataset from a CSV file (assuming 'your_data.csv' is in your Colab environment)
file_path = "your_data.csv"  # Update with your actual file path

# Load dataset using pandas
df = pd.read_csv(file_path)

# Check the loaded dataset
print(df.head())

# Convert the pandas DataFrame to a Hugging Face Dataset object
# Assuming your CSV has a 'text' column and a 'label' column
dataset = Dataset.from_pandas(df)

# Split the dataset into train, eval, and test
# Adjust these ranges or use a different splitting strategy as needed
train_size = int(0.8 * len(dataset))
eval_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - eval_size

train_dataset = dataset.select(range(train_size))
eval_dataset = dataset.select(range(train_size, train_size + eval_size))
test_dataset = dataset.select(range(train_size + eval_size, len(dataset)))

# Load a SetFit model from the Hub
model = SetFitModel.from_pretrained(
    "sentence-transformers/paraphrase-mpnet-base-v2",
    labels=train_dataset.unique('label')  # Dynamically get labels from your dataset
)

# Define training arguments
args = TrainingArguments(
    batch_size=16,
    num_epochs=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    metric="accuracy",
    column_mapping={"text": "text", "label": "label"}  # Ensure your column names are mapped correctly
)

# Train the model
trainer.train()

# Evaluate the model
metrics = trainer.evaluate(test_dataset)
print(metrics)

# Optional: Push model to the Hugging Face Hub
# trainer.push_to_hub("your-username/setfit-paraphrase-mpnet-base-v2-your-dataset")

# Run inference
preds = model.predict(["I loved the Spiderman movie!", "Pineapple on pizza is the worst 🤮"])
print(preds)
