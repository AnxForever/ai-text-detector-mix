import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

tokenizer = BertTokenizer.from_pretrained('models/bert_improved/best_model')
model = BertForSequenceClassification.from_pretrained('models/bert_improved/best_model').to(device)
model.eval()
print("✅ bert_improved 模型加载完成\n")

# 测试文本
test_texts = [
    {
        "name": "技术文档（AI生成）",
        "text": "梯度爆炸是指在训练神经网络时，反向传播得到的梯度在某些层或某些时间步变得非常大，导致参数更新幅度失控，训练不稳定甚至直接发散。常见表现：loss突然飙升、出现NaN；权重数值变得很大；训练过程极不稳定。"
    },
    {
        "name": "技术解释（AI生成）",
        "text": "这句话是偏圈内黑话的说法，大意是在讲：自己搭一个统一转发兼容层，把不同大模型伪装成同一种API，这样上层工具就能随便切换模型、随便用。"
    },
    {
        "name": "明显AI文本",
        "text": "作为一个AI语言模型，我可以为您提供以下建议。首先，我们需要考虑多个方面的因素。其次，这个问题需要从不同角度进行分析。最后，我建议您根据实际情况做出决策。"
    }
]

# 测试每个文本
for i, item in enumerate(test_texts, 1):
    print(f"{'='*60}")
    print(f"测试 {i}: {item['name']}")
    print(f"{'='*60}")
    
    text = item['text']
    encoding = tokenizer(
        text, 
        max_length=512, 
        padding='max_length',
        truncation=True, 
        return_tensors='pt'
    )
    
    with torch.no_grad():
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits[0], dim=0)
        pred = torch.argmax(outputs.logits[0]).item()
    
    prob_human = probs[0].item() * 100
    prob_ai = probs[1].item() * 100
    label = 'AI' if pred == 1 else 'Human'
    
    print(f"预测结果: {label}")
    print(f"置信度: {max(prob_human, prob_ai):.2f}%")
    print(f"人类概率: {prob_human:.2f}%")
    print(f"AI概率: {prob_ai:.2f}%")
    print(f"判断: {'✅ 正确' if label == 'AI' else '❌ 错误（应该是AI）'}")
    print()
