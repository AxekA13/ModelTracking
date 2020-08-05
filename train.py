import pytorch_lightning as pl
from torch import nn
import torch
import gc
import mlflow
import mlflow.pytorch
import pickle
import os
import yaml
import pandas as pd
import numpy as np
from git_commit import get_commit,get_commit_time
from model import EmoModel
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup,AutoModelWithLMHead,AdamW,BertTokenizer,BertForSequenceClassification
from dataset_preparation import EmoDataset,TokenizersCollateFn
from torch_lr_finder import LRFinder
from argparse import Namespace
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score,matthews_corrcoef
from typing import List
from functools import lru_cache
from dataset_preparation import label2int
from pytorch_lightning.metrics.classification import Accuracy,F1,Precision,Recall

## Methods required by PyTorchLightning
"""
    __init__ - to initialize custom variables in your model
    forward() - same as you would do in nn.Module - here I wrap my model inside of the pl.LightningModule but you could as well implement it directly here
    training_step() - take care of the forward pass for the model and return loss
    validation_step() and test_step()- same as above but for the different learning phases
    train_dataloader() - DataLoader for your training dataset
    val_dataloader() and test_dataloader() - same as above but for the different learning phases
    configure_optimizers() - prepare optimizer and (optionally) LR scheudles
    
    6 in EmoModel is len of emotions in datataset_preparation.py
"""

class TrainingModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased",num_labels=6)
        self.loss = nn.CrossEntropyLoss() ## combines LogSoftmax() and NLLLoss()
        self.hparams = hparams
        self.model_name = 'BERTa'
        self.save_hyperparameters()

    def step(self, batch, step_name="train"):
        b_input_ids, b_input_mask, b_labels = batch
        
        loss = self.loss(self.forward(b_input_ids,b_input_mask)[0],b_labels)
        loss_key = f"{step_name}_loss"
        tensorboard_logs = {loss_key: loss}

        mlflow.log_metric(("train_loss" if step_name == "train" else loss_key),loss.item())
        return { ("loss" if step_name == "train" else loss_key): loss, 'log': tensorboard_logs,
               "progress_bar": {loss_key: loss}}

    def forward(self, b_input_ids, attention_mask,token_type_ids=None,*args):
        return self.model(b_input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask,*args)

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")
    
    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def validation_end(self, outputs: List[dict]):
        loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        return {"val_loss": loss}
        
    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")
    
    def train_dataloader(self):
        return self.create_data_loader(self.hparams.train_path, shuffle=True)

    def val_dataloader(self):
        return self.create_data_loader(self.hparams.val_path)

    def test_dataloader(self):
        return self.create_data_loader(self.hparams.test_path)

                
    def create_data_loader(self, ds_path: str, shuffle=False):
        data = pd.read_csv(ds_path,delimiter=';',header=None,names = ['text','label'])
        sentences = data.text.values

        from sklearn.preprocessing import LabelEncoder
        labelencoder = LabelEncoder()
        data['label_enc'] = labelencoder.fit_transform(data['label'])
        data.rename(columns={'label':'label_desc'},inplace=True)
        data.rename(columns={'label_enc':'label'},inplace=True)

        MAX_LEN = 512
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)
        input_ids = [tokenizer.encode(sent, add_special_tokens=True,max_length=MAX_LEN,pad_to_max_length=True) for sent in sentences]
        labels = data.label.values
        
        attention_masks = []
        attention_masks = [[float(i>0) for i in seq]for seq in input_ids]


        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels)
        attention_masks = torch.tensor(attention_masks)
        print("Actual sentence before tokenization: ",sentences[2])
        print("Encoded Input from dataset: ",input_ids[2])
        print('MASK',attention_masks[2])
        print("Distribution of data based on labels: ",data.label.value_counts())

        dataset = torch.utils.data.TensorDataset(input_ids,attention_masks,labels)
        

        return DataLoader(
                    dataset,
                    batch_size=self.hparams.batch_size,
                    shuffle=shuffle)
        
    @lru_cache()
    def total_steps(self):
        return len(self.train_dataloader()) // self.hparams.accumulate_grad_batches * self.hparams.epochs

    def configure_optimizers(self):
        ## use AdamW optimizer -- faster approach to training NNs
        ## read: https://www.fast.ai/2018/07/02/adam-weight-decay/
        optimizer = AdamW(self.model.parameters(), lr=self.hparams.lr)
        lr_scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=self.hparams.warmup_steps,
                    num_training_steps=self.total_steps(),
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

# Function below aims to obtain valuable information about the optimal learning rate during a pretraining run.
# Determine boundary and increase the leanring rate linearly or exponentially.
# More: https://github.com/davidtvs/pytorch-lr-finder
def learningrate_finder(uper_bound,lower_bound,dataset_directory,end_learning =100,num_iterations=100):
    hparams_tmp = Namespace(
    train_path=dataset_directory + '/train.txt',
    val_path=dataset_directory + '/val.txt',
    test_path=dataset_directory + '/test.txt',
    batch_size=16,
    warmup_steps=100,
    epochs=1,
    lr= uper_bound,
    accumulate_grad_batches=1,)
    module = TrainingModule(hparams_tmp)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(module.parameters(), lr=lower_bound) ## lower bound LR
    lr_finder = LRFinder(module, optimizer, criterion, device="gpu")
    lr_finder.range_test(module.train_dataloader(), end_lr=end_learning, num_iter=num_iterations, accumulation_steps=hparams_tmp.accumulate_grad_batches)
    lr_finder.plot()
    #lr_finer.plot(show_lr=lr) show using learning rate
    lr_finder.reset()

if __name__ == '__main__':
    
    with open("params.yaml", 'r') as fd:
        params = yaml.safe_load(fd)
        
    hparams = Namespace(
    train_path=params['train']['train_path'],
    val_path=params['train']['val_path'],
    test_path=params['train']['test_path'],
    batch_size=params['train']['batch_size'],
    warmup_steps=params['train']['warmup_steps'],
    epochs=params['train']['epochs'],
    lr=float(params['train']['lr']),
    accumulate_grad_batches=params['train']['accumulate_grad_batches'])


    module = TrainingModule(hparams)

    ## garbage collection
    gc.collect()
    torch.cuda.empty_cache()

    mlflow.set_tracking_uri('file-plugin:/content/NLP_Emotions/mlruns')
    mlflow.set_experiment('LR Finder')
    acc = Accuracy()
    f1 = F1()
    precision = Precision()
    recall = Recall()
    with mlflow.start_run():
    ## train roughly for about 10-15 minutes with GPU enabled.
            trainer = pl.Trainer(gpus=1, max_epochs=hparams.epochs, progress_bar_refresh_rate=10,
                            accumulate_grad_batches=hparams.accumulate_grad_batches)
            trainer.fit(module)
            trainer.save_checkpoint('/content/NLP_Emotions/model/model.ckp')
            mlflow.log_params({'batch_size':hparams.batch_size,'warmup_steps':hparams.warmup_steps,'epochs':hparams.epochs,'learning_rate':hparams.lr,'accumulate_grad_batches':hparams.accumulate_grad_batches})

    # evaluating the trained model

            
            eval_accuracy,eval_mcc_accuracy,nb_eval_steps = 0, 0, 0
            module.eval()
            true_y,pred_y = [], []
            for i, batch_ in enumerate(module.val_dataloader()):

            # Unpack the inputs from our dataloader
                batch_ = tuple(t.to('cuda') for t in batch_)
                b_input_ids, b_input_mask, b_labels = batch_
            # Telling the model not to compute or store gradients, saving memory and speeding up validation
                with torch.no_grad():
            # Forward pass, calculate logit predictions
                    logits = module(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            
            # Move logits and labels to CPU
                logits = logits[0].to('cpu').numpy()
                label_ids = b_labels.to('cpu').numpy()

                pred_flat = np.argmax(logits, axis=1).flatten()
                labels_flat = label_ids.flatten()
                pred_y.extend(pred_flat)
                true_y.extend(labels_flat)
                
                df_metrics=pd.DataFrame({'Epoch':module.hparams.epochs,'Actual_class':labels_flat,'Predicted_class':pred_flat})
                
                tmp_eval_accuracy = accuracy_score(labels_flat,pred_flat)
                tmp_eval_mcc_accuracy = matthews_corrcoef(labels_flat, pred_flat)
                
                eval_accuracy += tmp_eval_accuracy
                eval_mcc_accuracy += tmp_eval_mcc_accuracy
                nb_eval_steps += 1

            print(F'\n\tValidation Accuracy: {eval_accuracy/nb_eval_steps}')
            print(F'\n\tValidation MCC Accuracy: {eval_mcc_accuracy/nb_eval_steps}')
            mlflow.log_metrics({'accuracy':acc(torch.tensor(pred_y),torch.tensor(true_y)).item(),'f1':f1(torch.tensor(pred_y),torch.tensor(true_y)).item(),'precision':precision(torch.tensor(pred_y),torch.tensor(true_y)).item(),'recall':recall(torch.tensor(pred_y),torch.tensor(true_y)).item()})
            mlflow.set_tag('Version','LRFinder')
            mlflow.set_tag('Stage','train')
            mlflow.set_tag('Commit', get_commit())
            mlflow.set_tag('Time',get_commit_time())
            mlflow.set_tag('Model',module.model_name)
           