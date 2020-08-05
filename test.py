import pickle
import yaml
import torch
import mlflow
import numpy as np
import subprocess
import pandas as pd
from train import TrainingModule
from git_commit import get_commit,get_commit_time
from sklearn.metrics import accuracy_score,matthews_corrcoef
from pytorch_lightning.metrics.classification import Accuracy,F1,Precision,Recall

def get_closest_gittag():
    commit = subprocess.check_output("git rev-parse HEAD",shell=True)
    tag = subprocess.check_output("git describe --abbrev=0 --tags " + str(commit)[2:-3],shell=True)
    return tag

if __name__ == '__main__':

    with open("params.yaml", 'r') as fd:
        params = yaml.safe_load(fd)

    MODEL_PATH = params['test']['model_path']

    module = TrainingModule.load_from_checkpoint(MODEL_PATH).to('cuda')
    module.eval()

    acc = Accuracy()
    f1 = F1()
    precision = Precision()
    recall = Recall()

    
    mlflow.set_tracking_uri('file-plugin:/content/NLP_Emotions/mlruns')
    if get_closest_gittag() == 'v2.0':
        mlflow.set_experiment('SGD')
        mlflow.set_tag('Version','SGD')
        mlflow.set_tag('Stage','test')
        mlflow.set_tag('Commit', get_commit())
        mlflow.set_tag('Time',get_commit_time())
        mlflow.set_tag('Model',module.model_name)
        mlflow.log_params({'batch_size':module.hparams.batch_size,'epochs':module.hparams.epochs,'learning_rate':module.hparams.lr})
    else:
        try:
            print(module.hparams.warmup_steps)
            mlflow.set_experiment('LR Finder')
            mlflow.set_tag('Version','LR Finder')
            mlflow.set_tag('Stage','test')
            mlflow.set_tag('Commit', get_commit())
            mlflow.set_tag('Time',get_commit_time())
            mlflow.set_tag('Model',module.model_name)
            mlflow.log_params({'batch_size':module.hparams.batch_size,'warmup_steps':module.hparams.warmup_steps,'epochs':module.hparams.epochs,'learning_rate':module.hparams.lr,'accumulate_grad_batches':module.hparams.accumulate_grad_batches})
        except AttributeError:
            mlflow.set_experiment('SGD')
            mlflow.set_tag('Version','SGD')
            mlflow.set_tag('Stage','test')
            mlflow.set_tag('Commit', get_commit())
            mlflow.set_tag('Time',get_commit_time())
            mlflow.set_tag('Model',module.model_name)
            mlflow.log_params({'batch_size':module.hparams.batch_size,'epochs':module.hparams.epochs,'learning_rate':module.hparams.lr})


    eval_accuracy,eval_mcc_accuracy,nb_eval_steps = 0, 0, 0
    pred_y, true_y = [],[]
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