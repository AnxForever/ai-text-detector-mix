import torch
from transformers import BertTokenizer, BertForSequenceClassification

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 测试文本
test_texts = [
    ("技术文档-AI生成", "梯度爆炸是指在训练神经网络时，反向传播得到的梯度在某些层或某些时间步变得非常大，导致参数更新幅度失控，训练不稳定甚至直接发散。常见表现：loss突然飙升、出现NaN；权重数值变得很大；训练过程极不稳定。"),
    ("技术解释-AI生成", "这句话是偏圈内黑话的说法，大意是在讲：自己搭一个统一转发兼容层，把不同大模型伪装成同一种API，这样上层工具就能随便切换模型、随便用。"),
    ("明显AI", "作为一个AI语言模型，我可以为您提供以下建议。首先，我们需要考虑多个方面的因素。其次，这个问题需要从不同角度进行分析。"),
]

models = [
    ("bert_v2_with_sep", "models/bert_v2_with_sep"),
    ("bert_improved/best", "models/bert_improved/best_model"),
]

for model_name, model_path in models:
    print(f"\n{'='*70}")
    print(f"测试模型: {model_name}")
    print(f"{'='*70}")
    
    try:
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path).to(device)
        model.eval()
        print(f"✅ 模型加载成功\n")
        
        for name, text in test_texts:
            encoding = tokenizer(text, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
            
            with torch.no_grad():
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits[0], dim=0)
                pred = torch.argmax(outputs.logits[0]).item()
            
            prob_human = probs[0].item() * 100
            prob_ai = probs[1].item() * 100
            label = 'AI' if pred == 1 else 'Human'
            
            result = '✅' if label == 'AI' else '❌'
            print(f"{result} {name:20s} → {label:6s} (AI:{prob_ai:5.1f}% Human:{prob_human:5.1f}%)")
            
    except Exception as e:
        print(f"❌ 加载失败: {e}")
