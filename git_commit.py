import subprocess

def get_commit():
    commit = subprocess.check_output("git rev-parse HEAD",shell=True)
    return str(commit)[2:9]

def get_commit_time():
    time = subprocess.check_output("git show -s --format=%ci HEAD",shell=True)
    return str(time)[2:21]