"""Install a hash-checked v112 transfer without replacing different existing files."""
import argparse,hashlib,json,tarfile
from pathlib import Path

def install(archive,root,expected):
    archive=Path(archive);root=Path(root).resolve()
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=expected:raise RuntimeError('transfer_hash_mismatch')
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers()
        hashes=json.load(tar.extractfile('reports/v112-training-transfer.json'))
        if {m.name for m in members}!=set(hashes)|{'reports/v112-training-transfer.json'}:raise RuntimeError('unexpected_member')
        for member in members:
            target=(root/member.name).resolve()
            if not member.isfile() or not target.is_relative_to(root):raise RuntimeError('unsafe_member')
            data=tar.extractfile(member).read()
            if member.name in hashes and hashlib.sha256(data).hexdigest()!=hashes[member.name]:raise RuntimeError('payload_mismatch')
            if target.exists() and target.read_bytes()!=data:raise RuntimeError('refuse_existing_different_file:'+member.name)
        for member in members:
            target=root/member.name
            if not target.exists():
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(tar.extractfile(member).read())
    print(json.dumps({'installed_files':len(members),'archive_sha256':expected}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--root',required=True);p.add_argument('--sha256',required=True);a=p.parse_args();install(a.archive,a.root,a.sha256)
