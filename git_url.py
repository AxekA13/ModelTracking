import subprocess

def get_commit_url():
    remote_url = subprocess.check_output("git remote get-url origin",shell=True)
    commit = subprocess.check_output("git rev-parse HEAD",shell=True)
    return str(remote_url).split('.git')[0][2:] + '/tree/' + str(commit)[2:-3]

def get_commit_time():
    time = subprocess.check_output("git show -s --format=%ci HEAD",shell=True)
    return str(time)[2:21]
print(get_commit_time())