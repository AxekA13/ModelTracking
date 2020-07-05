from pretrained_example import tokenizer
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
import argparse
import torch
import os
import pandas as pd
from torch.utils.data import Dataset
# Implement CollateFN using fast tokenizers. This function basically takes care of proper tokenization and batches of sequences

# emotion labels
label2int = {
"sadness": 0,
"joy": 1,
"love": 2,
"anger": 3,
"fear": 4,
"surprise": 5
}
class TokenizersCollateFn:
    def __init__(self, max_tokens=512):

        ## RoBERTa uses BPE tokenizer similar to GPT
        t = ByteLevelBPETokenizer(
            "tokenizer/vocab.json",
            "tokenizer/merges.txt"
        )
        t._tokenizer.post_processor = BertProcessing(
            ("</s>", t.token_to_id("</s>")),
            ("<s>", t.token_to_id("<s>")),
        )
        t.enable_truncation(max_tokens)
        t.enable_padding(length=max_tokens, pad_id=t.token_to_id("<pad>"))
        self.tokenizer = t

    def __call__(self, batch):
        encoded = self.tokenizer.encode_batch([x[0] for x in batch])
        sequences_padded = torch.tensor([enc.ids for enc in encoded])
        attention_masks_padded = torch.tensor([enc.attention_mask for enc in encoded])
        labels = torch.tensor([x[1] for x in batch])
        
        return (sequences_padded, attention_masks_padded), labels

# Create the Dataset object that will be used to load the different datasets.
class EmoDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.data_column = "text"
        self.class_column = "class"
        self.data = pd.read_csv(path, sep=";", header=None, names=[self.data_column, self.class_column],
                               engine="python")

    def __getitem__(self, idx):
        return self.data.loc[idx, self.data_column], label2int[self.data.loc[idx, self.class_column]]

    def __len__(self):
        return self.data.shape[0]

if __name__ == '__main__':

    
    path = 'tokenizer'

    try:
        os.makedirs(path,exist_ok=True)
    except OSError:
        print('Создать директорию %s не удалось' % path)
    else:
        print('Успешно создана директория %s' % path)

    # load pretrained tokenizer information
    tokenizer.save_pretrained("tokenizer")
    # check catalog
    print('Содержание каталога')
    os.system('ls tokenizer')

    # create dataset directory
    try:    
        os.mkdir('dataset')
    except OSError:
        print('Создать директорию %s не удалось' % path)
    else:
        print('Успешно создана директория %s' % path)
    
    # parser command line
    parser = argparse.ArgumentParser(description='Dataset preparation. Use --s for split dataset')
    parser.add_argument('-s','--split',help='Split dataset',action='store_true')
    args = parser.parse_args()
    
    if args.split:
            os.system('wget https://www.dropbox.com/s/607ptdakxuh5i4s/merged_training.pkl')
            import pickle
            # helper function
            def load_from_pickle(directory):
                return pickle.load(open(directory,"rb"))

            ## export the datasets as txt files
            ## EXERCISE: Change this to an address    
            train_path = "dataset/train.txt"
            test_path = "dataset/test.txt"
            val_path = "dataset/val.txt"

            data = load_from_pickle(directory="merged_training.pkl")
            # using a sample
            emotions = [ "sadness", "joy", "love", "anger", "fear", "surprise"]
            data= data[data["emotions"].isin(emotions)]

            data = data.sample(n=20000);
            data.emotions.value_counts().plot.bar()

            print(data.count())
            print(data.head())

            # reset index
            data.reset_index(drop=True, inplace=True)
            
            # check unique emotions in the dataset
            print(data.emotions.unique())


            # Split the data and store into individual text files
    
            from sklearn.model_selection import train_test_split
            import numpy as np

            # Creating training and validation sets using an 80-20 split
            input_train, input_val, target_train, target_val = train_test_split(data.text.to_numpy(), 
                                                                                data.emotions.to_numpy(), 
                                                                                test_size=0.2)

            # Split the validataion further to obtain a holdout dataset (for testing) -- split 50:50
            input_val, input_test, target_val, target_test = train_test_split(input_val, target_val, test_size=0.5)


            ## create a dataframe for each dataset
            train_dataset = pd.DataFrame(data={"text": input_train, "class": target_train})
            val_dataset = pd.DataFrame(data={"text": input_val, "class": target_val})
            test_dataset = pd.DataFrame(data={"text": input_test, "class": target_test})
            final_dataset = {"train": train_dataset, "val": val_dataset , "test": test_dataset }

            train_dataset.to_csv(train_path, sep=";",header=False, index=False)
            val_dataset.to_csv(test_path, sep=";",header=False, index=False)
            test_dataset.to_csv(val_path, sep=";",header=False, index=False)

            # Sanity check
            ds = EmoDataset(train_path)
            ds[19]
            
    else: 
        os.system('wget -P dataset https://www.dropbox.com/s/ikkqxfdbdec3fuj/test.txt & wget -P dataset https://www.dropbox.com/s/1pzkadrvffbqw6o/train.txt & wget -P dataset https://www.dropbox.com/s/2mzialpsgf9k5l3/val.txt')