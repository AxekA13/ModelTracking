import shutil
import os
import subprocess as sbp

def merge_mlurns(downloaded_mlruns_path='/home/axeka/Загрузки/mlruns',local_mlruns_path='/home/axeka/VSCodeProjects/NLP_Emotions/NLP_Emotions/mlruns'):

    fol = os.listdir(downloaded_mlruns_path)
    for i in fol:
        p1 = os.path.join(downloaded_mlruns_path,i)
        p3 = 'cp -r ' + p1 +' ' + local_mlruns_path +'/.'
        sbp.Popen(p3,shell=True)

if __name__ == '__main__':

# move mlurns to google colab
source = '/content/NLP/mlruns'
dest = '/colabdrive/mlruns/'

shutil.copytree(source,dest)

