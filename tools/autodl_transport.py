"""Local SSH transport. Payloads stream to AutoDL/GitHub, never to local files."""
import argparse
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def ssh(command):
    return ['ssh', '-p', '34903', '-i', str(ROOT / 'tmp/autodl/id_ed25519'),
            '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
            '-o', 'UserKnownHostsFile=tmp/autodl/known_hosts',
            '-o', 'ServerAliveInterval=30', 'root@connect.bjb1.seetacloud.com', command]


def run(command, **kwargs):
    return subprocess.run(ssh(command), cwd=ROOT, check=True, **kwargs)


def upload(paths):
    # Separate argv avoids PowerShell quoting and binary-pipe corruption.
    producer = subprocess.Popen(['tar', '--exclude=__pycache__', '-cf', '-', *paths], cwd=ROOT, stdout=subprocess.PIPE)
    try:
        run('tar -m -xf - -C /root/autodl-tmp/arp', stdin=producer.stdout)
    finally:
        producer.stdout.close()
    if producer.wait():
        raise RuntimeError('Local archive failed')


def credential():
    # Credential helper output is deliberately captured and never logged/stored.
    env = {**os.environ, 'GIT_TERMINAL_PROMPT': '0', 'GCM_INTERACTIVE': 'never'}
    result = subprocess.run(['git', 'credential', 'fill'], input='protocol=https\nhost=github.com\n\n',
                            text=True, capture_output=True, check=True, cwd=ROOT, env=env)
    fields = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    if not fields.get('password'):
        raise RuntimeError('No GitHub credential available')
    return fields['password']


def publish(remote_command):
    # The credential travels only through the authenticated encrypted SSH stdin.
    run(remote_command, input=(credential() + '\n').encode())


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['run', 'upload', 'upload-lock', 'publish'])
    p.add_argument('args', nargs='*')
    a = p.parse_args()
    if a.action == 'upload':
        upload(a.args)
    elif a.action == 'upload-lock':
        import json
        lock = json.loads((ROOT / 'data/experiment.lock.json').read_text(encoding='utf-8'))
        upload([*lock['files'], 'data/experiment.lock.json'])
    elif a.action == 'publish':
        publish(' '.join(a.args))
    else:
        run(' '.join(a.args))
