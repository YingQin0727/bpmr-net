"""Synthetic permission tests, never scientific payloads."""
import hashlib
import tempfile
from pathlib import Path
from paper_revision.replication_firewall import verified_artifact


def test_role_identity_and_path_guards():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder) / 'staged'
        root.mkdir()
        (root / 'synthetic').write_bytes(b'non-scientific')
        record = dict(identity='REVISION_REPLICATION', split='train', path='synthetic',
                      sha256=hashlib.sha256(b'non-scientific').hexdigest())
        assert verified_artifact(root, record, 'train').name == 'synthetic'
        (Path(folder) / 'outside').write_bytes(b'non-scientific')
        for change, role in [({'split':'test_truth'}, 'train'), ({'split':'validation'}, 'reconstruct'),
                             ({'identity':'HISTORICAL'}, 'train'), ({'path':'../outside'}, 'train'),
                             ({'sha256':'0'*64}, 'train')]:
            try:
                verified_artifact(root, record | change, role)
            except PermissionError:
                continue
            raise AssertionError((change, role))
