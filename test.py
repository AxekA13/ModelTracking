import pickle
import yaml
import torch
import mlflow
import subprocess
from train import TrainingModule
from git_commit import get_commit,get_commit_time
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
            

    with torch.no_grad():
        true_y, pred_y = [],[]
        for i, batch_ in enumerate(module.val_dataloader()):
            (X, attn), y = batch_
            batch = (X.cuda(), attn.cuda())
            y_pred = torch.argmax(module(batch), dim=1)
            true_y.extend(y)
            pred_y.extend(y_pred)
        
    mlflow.log_metrics({'accuracy':acc(torch.tensor(pred_y),torch.tensor(true_y)).item(),'f1':f1(torch.tensor(pred_y),torch.tensor(true_y)).item(),'precision':precision(torch.tensor(pred_y),torch.tensor(true_y)).item(),'recall':recall(torch.tensor(pred_y),torch.tensor(true_y)).item()})