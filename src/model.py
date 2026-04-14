import torch
from transformers import BertForSequenceClassification
from functools import partial

class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        return self.alpha * (x @ self.A @ self.B)

class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x):
        return self.linear(x) + self.lora(x)

def build_lora_model(device, lora_rank=8, lora_alpha=16):
    """Initializes BERT, freezes base weights, and injects LoRA layers."""
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False
    )

    for param in model.parameters():
        param.requires_grad = False

    assign_lora = partial(LinearWithLoRA, rank=lora_rank, alpha=lora_alpha)
    for layer in model.bert.encoder.layer:
        layer.attention.self.query = assign_lora(layer.attention.self.query)
        layer.attention.self.value = assign_lora(layer.attention.self.value)

    for param in model.classifier.parameters():
        param.requires_grad = True

    model.to(device)
    return model