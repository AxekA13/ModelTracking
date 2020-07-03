import torch
from torch import nn
import torch.nn.functional as F
# activation function from https://github.com/digantamisra98/Mish/blob/b5f006660ac0b4c46e2c6958ad0301d7f9c59651/Mish/Torch/mish.py
@torch.jit.script
def mish(input):
    return input * torch.tanh(F.softplus(input))
  
class Mish(nn.Module):
    def forward(self, input):
        return mish(input)

# model
class EmoModel(nn.Module):
    def __init__(self, base_model, n_classes, base_model_output_size=768, dropout=0.05):
        super().__init__()
        self.base_model = base_model
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(base_model_output_size, base_model_output_size),
            Mish(),
            nn.Dropout(dropout),
            nn.Linear(base_model_output_size, n_classes)
        )
        
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(mean=0.0, std=0.02)
                if layer.bias is not None:
                    layer.bias.data.zero_()

    def forward(self, input_, *args):
        X, attention_mask = input_
        hidden_states = self.base_model(X, attention_mask=attention_mask)
        
        # maybe do some pooling / RNNs... go crazy here!
        
        # use the <s> representation
        return self.classifier(hidden_states[0][:, 0, :])


if __name__ == '__main__':
    from transformers import AutoTokenizer, AutoModelWithLMHead
    import argparse
    parser = argparse.ArgumentParser(description='Pretest the model with dummy text')

    # load tokenizer for this model
    tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')

    # We want to ensure that the model is returing the right information back.
    classifier = EmoModel(AutoModelWithLMHead.from_pretrained("distilroberta-base").base_model, 3)
    t = "Elvis is the king of rock"
    enc = tokenizer.encode_plus(t)
    X = torch.tensor(enc["input_ids"]).unsqueeze(0).to('cpu')
    attn = torch.tensor(enc["attention_mask"]).unsqueeze(0).to('cpu')

    # check output. Correct output is  3-class output (for CrossEntropyLoss).
    print(classifier((X, attn)))
    