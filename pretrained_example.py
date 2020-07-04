import torch
from transformers import AutoTokenizer, AutoModelWithLMHead

# load the tokenizer for model and load model withou head
tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')


if __name__ == '__main__':
    model = AutoModelWithLMHead.from_pretrained("distilroberta-base")
    base_model = model.base_model

    #check out the tokenizer
    text = "Elvis is the king of rock!"
    enc = tokenizer.encode_plus(text)
    enc.keys()
    print(enc)

    # sequence_output, pooled_output, (hidden_states), (attentions)
    out = base_model(torch.tensor(enc["input_ids"]).unsqueeze(0), torch.tensor(enc["attention_mask"]).unsqueeze(0))
    print(out[0].shape) #torch.Size([1, 9,768]) represents batch_size, number of tokens in input text (lenght of tokenized text), model's output hidden size.


    # This dimension contains internal representation of each of the input tokens
    # Representation of each token it is for example [0,355]. 0 for token_representation[0], 355 for token_representation[1] etc.
    t = "Elvis is the king of rock"
    enc = tokenizer.encode_plus(t)
    token_representations = base_model(torch.tensor(enc["input_ids"]).unsqueeze(0))[0][0]
    print(enc["input_ids"])
    print(tokenizer.decode(enc["input_ids"]))
    print(f"Length: {len(enc['input_ids'])}")
    print(token_representations.shape)
