import subprocess

def get_commit_url():
    remote_url = subprocess.check_output("git remote get-url origin",shell=True)
    commit = subprocess.check_output("git rev-parse HEAD",shell=True)
    return str(remote_url).split('.git')[0][2:] + '/tree/' + str(commit)[2:-3]
