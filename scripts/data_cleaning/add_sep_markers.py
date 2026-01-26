#!/usr/bin/env python3
import pandas as pd
import json

# Load expanded hybrid dataset
df = pd.read_csv('datasets/hybrid/hybrid_dataset_expanded.csv')
print(f"Total samples: {len(df)}")
print(df['category'].value_counts())

# Load all C2 data with boundary info
import glob
c2_files = glob.glob('datasets/hybrid/c2*.json')
print(f"Found C2 files: {len(c2_files)}")

boundary_map = {}
for file in c2_files:
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            if 'boundary' in item:
                boundary_map[item['text']] = item['boundary']

print(f"\nBoundary info available: {len(boundary_map)}")

# Add [SEP] to C2 samples
c2_df = df[df['category'] == 'C2'].copy()
updated = 0
no_boundary = 0

new_texts = []
for idx, row in c2_df.iterrows():
    text = row['text']
    if text in boundary_map:
        boundary = boundary_map[text]
        new_text = text[:boundary] + '[SEP]' + text[boundary:]
        df.at[idx, 'text'] = new_text
        updated += 1
    else:
        no_boundary += 1
    
print(f"\nUpdated with [SEP]: {updated}/{len(c2_df)}")
print(f"No boundary info: {no_boundary}/{len(c2_df)}")

# Save updated dataset
df.to_csv('datasets/hybrid/hybrid_dataset_with_sep.csv', index=False)
print(f"\nSaved to: datasets/hybrid/hybrid_dataset_with_sep.csv")

# Verify
sample = df[df['category'] == 'C2'].iloc[0]['text']
if '[SEP]' in sample:
    parts = sample.split('[SEP]')
    print(f"\nVerification - Sample C2:")
    print(f"  Human part: {len(parts[0])} chars")
    print(f"  AI part: {len(parts[1])} chars")
    print(f"  Preview: {parts[0][:80]}...[SEP]{parts[1][:80]}...")
