import pytorch_lightning as pl
from torch import nn
import torch
import gc
import mlflow
import mlflow.pytorch
import pickle
import os
import yaml
from git_commit import get_commit,get_commit_time
from model import EmoModel
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup,AutoModelWithLMHead,AdamW
from dataset_preparation import EmoDataset,TokenizersCollateFn
from torch_lr_finder import LRFinder
from argparse import Namespace
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score  
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
        self.model = EmoModel(AutoModelWithLMHead.from_pretrained("distilroberta-base").base_model,6)
        self.loss = nn.CrossEntropyLoss() ## combines LogSoftmax() and NLLLoss()
        self.hparams = hparams
        self.model_name = 'ROBERTa'
        self.save_hyperparameters()

    def step(self, batch, step_name="train"):
        X, y = batch
        loss = self.loss(self.forward(X), y)
        loss_key = f"{step_name}_loss"
        tensorboard_logs = {loss_key: loss}

        mlflow.log_metric(("train_loss" if step_name == "train" else loss_key),loss.item())
        return { ("loss" if step_name == "train" else loss_key): loss, 'log': tensorboard_logs,
               "progress_bar": {loss_key: loss}}

    def forward(self, X, *args):
        return self.model(X, *args)

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
        return DataLoader(
                    EmoDataset(ds_path),
                    batch_size=self.hparams.batch_size,
                    shuffle=shuffle,
                    collate_fn=TokenizersCollateFn()
        )

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hparams.lr)
        return optimizer


if __name__ == '__main__':
    
    with open("params.yaml", 'r') as fd:
        params = yaml.safe_load(fd)
        
    hparams = Namespace(
    train_path=params['train']['train_path'],
    val_path=params['train']['val_path'],
    test_path=params['train']['test_path'],
    batch_size=params['train']['batch_size'],
    epochs=params['train']['epochs'],
    lr=float(params['train']['lr']))


    module = TrainingModule(hparams)

    ## garbage collection
    gc.collect()
    torch.cuda.empty_cache()

    mlflow.set_tracking_uri('file-plugin:/content/NLP_Emotions/mlruns')
    mlflow.set_experiment('SGD')
    acc = Accuracy()
    f1 = F1()
    precision = Precision()
    recall = Recall()
    with mlflow.start_run():
    ## train roughly for about 10-15 minutes with GPU enabled.
            trainer = pl.Trainer(gpus=1, max_epochs=hparams.epochs, progress_bar_refresh_rate=10)
            trainer.fit(module)
            trainer.save_checkpoint('/content/NLP_Emotions/model/model.ckp')
            mlflow.log_params({'batch_size':hparams.batch_size,'epochs':hparams.epochs,'learning_rate':hparams.lr})

    # evaluating the trained model

            
            with torch.no_grad():
                progress = ["/", "-", "\\", "|", "/", "-", "\\", "|"]
                module.eval()
                true_y, pred_y = [], []
                for i, batch_ in enumerate(module.test_dataloader()):
                    (X, attn), y = batch_
                    batch = (X.cuda(), attn.cuda())
                    print(progress[i % len(progress)], end="\r")
                    y_pred = torch.argmax(module(batch), dim=1)
                    true_y.extend(y)
                    pred_y.extend(y_pred)
            mlflow.log_metrics({'accuracy':acc(torch.tensor(pred_y),torch.tensor(true_y)).item(),'f1':f1(torch.tensor(pred_y),torch.tensor(true_y)).item(),'precision':precision(torch.tensor(pred_y),torch.tensor(true_y)).item(),'recall':recall(torch.tensor(pred_y),torch.tensor(true_y)).item()})
            mlflow.set_tag('Version','SGD')
            mlflow.set_tag('Stage','train')
            mlflow.set_tag('Commit', get_commit())
            mlflow.set_tag('Time',get_commit_time())
            mlflow.set_tag('Model',module.model_name)
           