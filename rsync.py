import os
import argparse

parser = argparse.ArgumentParser(description='Rsync argument parser')
parser.add_argument('-sf','--source_folder',help='source synchronizing folder',default='root@127.0.0.1:/colabdrive/mlruns/')
parser.add_argument('-tf','--target_folder',help='target synchronizing folder',default='/home/axeka/VSCodeProjects/NLP_Emotions/NLP_Emotions/mlruns')
parser.add_argument('-p','--port',help='port for connection',default='9999')
args = parser.parse_args()

print(f'rsync -zavP "-e ssh -p {args.port}"{args.source_folder} {args.target_folder}')
os.system(f'rsync -zaP "-e ssh -p {args.port}"root@127.0.0.1:{args.source_folder} {args.target_folder}')
os.system('mlflow ui')



