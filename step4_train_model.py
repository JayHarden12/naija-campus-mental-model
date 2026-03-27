import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import pickle

# 1. Load the labeled dataset
print("Loading Labeled Dataset...")
df = pd.read_csv('Master_Dataset_Labeled.csv')

# Drop rows where intents or the original context might be empty
df = df.dropna(subset=['intents', 'context'])

# 2. Prepare the Multi-Label Target Variable
print("Binarizing Intents...")
df['intent_list'] = df['intents'].apply(lambda x: [i.strip() for i in str(x).split(',')])

mlb = MultiLabelBinarizer()
labels_matrix = mlb.fit_transform(df['intent_list'])
num_labels = len(mlb.classes_)
print(f"Total Unique Intents Found: {num_labels}")

# 3. Prepare the Input Text
# Using the Pidgin column, falling back to English if missing
texts = df['naija_pidgin'].fillna(df['context']).tolist() 
labels = labels_matrix.astype(float).tolist()

# 4. The 60-20-20 Academic Split
print("Splitting dataset into 60% Train, 20% Validation, 20% Test...")

# First split: Pull out the 20% Test set
train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Second split: Divide the remaining 80% into 60% Train and 20% Validation (0.25 * 0.8 = 0.2)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_val_texts, train_val_labels, test_size=0.25, random_state=42
)

print(f"-> Training on {len(train_texts)} rows.")
print(f"-> Validating on {len(val_texts)} rows during training.")
print(f"-> Holding out {len(test_texts)} rows for final blind testing.")

# 5. Save the Holdout Test Set for your thesis evaluation later
test_df = pd.DataFrame({'text': test_texts, 'labels': test_labels})
test_df.to_csv('Holdout_Test_Set.csv', index=False)
print("Saved 'Holdout_Test_Set.csv' for post-training evaluation.")

# 6. Tokenization (Upgraded to 256 Tokens)
print("Tokenizing Text...")
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=256)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=256)

# 7. Create PyTorch Dataset
class NaijaDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = NaijaDataset(train_encodings, train_labels)
val_dataset = NaijaDataset(val_encodings, val_labels)

# 8. Load DistilBERT for Multi-Label Classification
print("Loading DistilBERT Model...")
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', 
    num_labels=num_labels,
    problem_type="multi_label_classification" 
)

# 9. Set Training Arguments (Optimized for 256 Tokens)
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=4,              
    per_device_train_batch_size=8,   # Halved to prevent 256-token OOM memory crash
    per_device_eval_batch_size=32,   
    warmup_steps=80,                 # Adjusted to 10% of new total steps
    weight_decay=0.01,               
    logging_dir='./logs',
    logging_steps=40,                # Prints updates smoothly
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

# 10. Train the Model
print("Starting Training...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

# 11. Save Everything
print("Saving Model and Label Maps...")
model.save_pretrained('./naija_campus_model')
tokenizer.save_pretrained('./naija_campus_model')

with open('./naija_campus_model/mlb_classes.pkl', 'wb') as f:
    pickle.dump(mlb.classes_, f)

print("✅ Model Training Complete! Remember to zip and download the folder.")