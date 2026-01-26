# AI Text Detection System - Agent Guidelines

## Build, Lint, Test Commands

### Environment Setup
```bash
# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements_training.txt
```

### Running Tests
```bash
# Run single text test (interactive mode)
python scripts/evaluation/test_single_text.py --interactive

# Complete evaluation on all test sets
python scripts/evaluation/eval_complete.py

# Run specific evaluation script
python scripts/evaluation/complete_evaluation.py

# Run demo visualization
python scripts/demo/visualize_detection.py

# Test V0 API integration
python test_v0_api.py
```

### Training Models
```bash
# Train BERT classifier with improvements
python scripts/training/train_bert_improved.py --epochs 5 --batch_size 16

# Train span detector for boundary detection
python scripts/training/train_span_detector.py --epochs 10

# Train BiGRU variant
python scripts/training/train_bert_bigru.py --epochs 5

# Train DPCNN variant
python scripts/training/train_dpcnn.py --epochs 5
```

### Data Processing
```bash
# Add SEP markers for hybrid text
python scripts/data_cleaning/add_sep_markers.py

# Prepare span labels
python scripts/data_cleaning/prepare_span_labels.py

# Rebuild combined dataset v2
python scripts/data_cleaning/rebuild_combined_v2.py
```

## Code Style Guidelines

### Imports
- Group imports: standard library, third-party, local modules
- Use absolute imports for local modules
- Add project root to sys.path at module start

```python
import os
import sys
import json

import torch
import pandas as pd
from transformers import BertTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from scripts.bert_prep.create_bert_dataset import AIDetectionDataset
```

### Formatting
- Max line length: 100 characters
- Use 4 spaces for indentation (no tabs)
- Add docstrings for all functions and classes
- Use type hints for function parameters and return values

```python
def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    epochs: int,
    learning_rate: float = 2e-5
) -> Dict[str, float]:
    """Train the model and return metrics.

    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer

    Returns:
        Dictionary containing training metrics
    """
    pass
```

### Naming Conventions
- **Variables**: snake_case (e.g., `max_length`, `batch_size`)
- **Functions**: snake_case (e.g., `train_model`, `evaluate_model`)
- **Classes**: PascalCase (e.g., `BERTTrainer`, `TextDataset`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MAX_LENGTH`, `BATCH_SIZE`)
- **Private methods**: _leading_underscore (e.g., `_prepare_data`)

### Error Handling
- Always wrap file operations in try-except blocks
- Use specific exceptions when possible
- Provide meaningful error messages

```python
try:
    model = BertForSequenceClassification.from_pretrained(model_path)
except FileNotFoundError as e:
    print(f"Model file not found: {model_path}")
    sys.exit(1)
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)
```

### Device Handling
- Always check CUDA availability
- Support both CPU and GPU
- Move tensors to device explicitly

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
input_ids = input_ids.to(device)
```

### Model Configuration
- Use argparse for command-line arguments
- Provide sensible defaults
- Log configuration at startup

```python
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='bert-base-chinese')
parser.add_argument('--max_length', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epochs', type=int, default=5)
args = parser.parse_args()
```

### Data Loading
- Use DataLoader for batch processing
- Implement custom Dataset classes
- Handle variable-length sequences with padding

```python
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }
```

### Training Loop
- Use tqdm for progress bars
- Save best model based on validation loss
- Log metrics after each epoch

```python
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_loss = train_loss / len(train_loader)
    print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')
```

### Evaluation Metrics
- Use sklearn.metrics for evaluation
- Report accuracy, precision, recall, F1-score
- Generate classification reports

```python
from sklearn.metrics import accuracy_score, classification_report

preds = model.predict(test_loader)
accuracy = accuracy_score(true_labels, preds)
report = classification_report(true_labels, preds, target_names=['Human', 'AI'])
print(f'Accuracy: {accuracy:.4f}')
print(report)
```

### File Organization
- Keep training scripts in `scripts/training/`
- Keep evaluation scripts in `scripts/evaluation/`
- Keep data processing in `scripts/data_cleaning/`
- Save models in `models/` directory
- Save logs in `logs/` directory

### Logging
- Use print statements for simple logging
- Include timestamps for important events
- Log both successes and failures

```python
import datetime
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"[{timestamp}] Starting training...")
print(f"[{timestamp}] Training completed successfully")
```

### Environment Variables
- Set HF_HUB_OFFLINE=1 for offline mode
- Set TRANSFORMERS_OFFLINE=1 to disable downloads

```python
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
```

### Model Saving/Loading
- Save model state_dict and tokenizer
- Include configuration in saved files
- Use torch.save with protocol 4

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'config': config_dict
}, f'{output_dir}/model.pt')

tokenizer.save_pretrained(f'{output_dir}/tokenizer')
```

### Chinese Text Handling
- Set PYTHONIOENCODING=utf-8
- Use UTF-8 encoding for file operations
- Handle Chinese characters correctly

```python
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

with open('data.txt', 'r', encoding='utf-8') as f:
    text = f.read()
```

### Best Practices
- Always validate input data
- Check for CUDA availability before using GPU
- Use gradient clipping to prevent exploding gradients
- Implement early stopping for training
- Save checkpoints regularly
- Test on small subset before full training
- Use random seeds for reproducibility

### Reproducibility
```python
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
```

### Common Patterns
- **Model initialization**: Load pretrained BERT with classification head
- **Data preprocessing**: Tokenize with padding and truncation
- **Training loop**: Forward pass, loss computation, backward pass, optimizer step
- **Evaluation**: Disable gradient computation, collect predictions, compute metrics
- **Model checkpointing**: Save best model based on validation performance