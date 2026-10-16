"""Transport adapter for the user-authorized migrated RTX 5090 instance."""
import autodl_transport as transport

_original = transport.ssh


def ssh(command):
    args = _original(command)
    args[2] = '20479'
    args[-2] = 'root@connect.bjb2.seetacloud.com'
    return args


transport.ssh = ssh

if __name__ == '__main__':
    import sys
    action, *arguments = sys.argv[1:]
    if action == 'upload':
        transport.upload(arguments)
    elif action == 'publish':
        transport.publish(' '.join(arguments))
    else:
        transport.run(' '.join(arguments))
