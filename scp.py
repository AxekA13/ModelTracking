import os
import argparse

parser = argparse.ArgumentParser(description='Rsync argument parser')
parser.add_argument('-sf','--source_folder',help='source synchronizing folder',default='root@2.tcp.ngrok.io:/content/NLP_Emotions/mlruns')
parser.add_argument('-tf','--target_folder',help='target synchronizing folder',default='/home/axeka/VSCodeProjects/NLP_Emotions/NLP_Emotions/mlruns')
parser.add_argument('-p','--port',help='port for connection',default='9999')
args = parser.parse_args()

print(f'scp -r -P {args.port} {args.source_folder} {args.target_folder}')
os.system(f'scp -r -P {args.port} {args.source_folder} {args.target_folder}')



