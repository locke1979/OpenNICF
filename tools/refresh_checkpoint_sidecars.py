from hashlib import sha256
from pathlib import Path
import os, tempfile

root = Path(__file__).resolve().parents[1] / 'evaluation/issue102-newcorpus-v1/checkpoints'
for name in ('documents.json', 'queries.json'):
    source = root / name
    digest = sha256(source.read_bytes()).hexdigest() + '  ' + name + '\n'
    fd, tmp = tempfile.mkstemp(prefix=name + '.', dir=root)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            handle.write(digest)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, root / (name + '.sha256'))
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
