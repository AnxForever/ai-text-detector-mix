#!/usr/bin/env python3
import pandas as pd
from sklearn.model_selection import train_test_split

# Load original data
original_train = pd.read_csv('datasets/final_clean/train.csv')
original_val = pd.read_csv('datasets/final_clean/val.csv')
original_test = pd.read_csv('datasets/final_clean/test.csv')
original_df = pd.concat([original_train, original_val, original_test], ignore_index=True)
print(f"Original data: {len(original_df)}")

# Load hybrid data with [SEP]
hybrid_df = pd.read_csv('datasets/hybrid/hybrid_dataset_with_sep.csv')
print(f"Hybrid data: {len(hybrid_df)}")
print(hybrid_df['category'].value_counts())

# Verify [SEP] in C2
c2_with_sep = hybrid_df[hybrid_df['category'] == 'C2']['text'].str.contains('\[SEP\]').sum()
print(f"\nC2 samples with [SEP]: {c2_with_sep}/{len(hybrid_df[hybrid_df['category'] == 'C2'])}")

# Sample human control
human_control = original_df[original_df['label'] == 0].sample(n=3000, random_state=42)
human_control['category'] = 'Human'

# Combine
all_data = pd.concat([original_df, hybrid_df, human_control], ignore_index=True)
print(f"\nTotal combined: {len(all_data)}")

# Split
train, temp = train_test_split(all_data, test_size=0.2, random_state=42, stratify=all_data['label'])
val, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp['label'])

print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")

# Save
train.to_csv('datasets/combined_v2/train.csv', index=False)
val.to_csv('datasets/combined_v2/val.csv', index=False)
test.to_csv('datasets/combined_v2/test.csv', index=False)

# Hybrid-only test
hybrid_test = test[test['category'].isin(['C2', 'C3', 'C4', 'Human'])]
hybrid_test.to_csv('datasets/combined_v2/test_hybrid_only.csv', index=False)

print(f"\nHybrid test: {len(hybrid_test)}")
print(hybrid_test['category'].value_counts())

# Verify C2 in test
c2_test = hybrid_test[hybrid_test['category'] == 'C2']
c2_test_sep = c2_test['text'].str.contains('\[SEP\]').sum()
print(f"\nC2 in test with [SEP]: {c2_test_sep}/{len(c2_test)}")

print("\nSaved to: datasets/combined_v2/")
