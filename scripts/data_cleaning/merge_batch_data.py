#!/usr/bin/env python3
import json
import pandas as pd
from sklearn.model_selection import train_test_split

# Load batch data
with open('datasets/hybrid/c2_batch.json') as f:
    c2_batch = json.load(f)
with open('datasets/hybrid/c3_batch.json') as f:
    c3_batch = json.load(f)
with open('datasets/hybrid/c4_batch.json') as f:
    c4_batch = json.load(f)

print(f"Loaded: C2={len(c2_batch)}, C3={len(c3_batch)}, C4={len(c4_batch)}")

# Load existing hybrid data
hybrid_df = pd.read_csv('datasets/hybrid/hybrid_dataset.csv')
print(f"Existing hybrid data: {len(hybrid_df)}")

# Convert batch to dataframe
new_data = []
for item in c2_batch:
    new_data.append({'text': item['text'], 'label': 1, 'category': 'C2'})
for item in c3_batch:
    new_data.append({'text': item['text'], 'label': 1, 'category': 'C3'})
for item in c4_batch:
    new_data.append({'text': item['text'], 'label': 1, 'category': 'C4'})

new_df = pd.DataFrame(new_data)
print(f"New batch data: {len(new_df)}")

# Combine
combined_hybrid = pd.concat([hybrid_df, new_df], ignore_index=True)
print(f"Total hybrid data: {len(combined_hybrid)}")
print(combined_hybrid['category'].value_counts())

# Save expanded hybrid dataset
combined_hybrid.to_csv('datasets/hybrid/hybrid_dataset_expanded.csv', index=False)
print("\nSaved to: datasets/hybrid/hybrid_dataset_expanded.csv")

# Recreate combined dataset
original_train = pd.read_csv('datasets/final_clean/train.csv')
original_val = pd.read_csv('datasets/final_clean/val.csv')
original_test = pd.read_csv('datasets/final_clean/test.csv')
original_df = pd.concat([original_train, original_val, original_test], ignore_index=True)
print(f"\nOriginal data: {len(original_df)}")

# Sample human control
human_control = original_df[original_df['label'] == 0].sample(n=3000, random_state=42)
human_control['category'] = 'Human'

# Combine all
all_data = pd.concat([original_df, combined_hybrid, human_control], ignore_index=True)
print(f"Total combined: {len(all_data)}")

# Split
train, temp = train_test_split(all_data, test_size=0.2, random_state=42, stratify=all_data['label'])
val, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp['label'])

print(f"\nSplit: train={len(train)}, val={len(val)}, test={len(test)}")

# Save
train.to_csv('datasets/combined_v2/train.csv', index=False)
val.to_csv('datasets/combined_v2/val.csv', index=False)
test.to_csv('datasets/combined_v2/test.csv', index=False)

# Create hybrid-only test
hybrid_test = test[test['category'].isin(['C2', 'C3', 'C4', 'Human'])]
hybrid_test.to_csv('datasets/combined_v2/test_hybrid_only.csv', index=False)

print(f"\nHybrid test: {len(hybrid_test)}")
print(hybrid_test['category'].value_counts())
print("\nSaved to: datasets/combined_v2/")
