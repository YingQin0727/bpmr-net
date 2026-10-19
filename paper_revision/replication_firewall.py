"""REVISION_REPLICATION role-scoped staged-artifact access (never raw data)."""
import hashlib
from pathlib import Path

ALLOWED = {'train': {'train'}, 'select': {'validation'},
           'reconstruct': {'observations'}, 'evaluate': {'test_truth', 'predictions'}}


def verified_artifact(root, record, role):
    """Fail closed on identity, role, traversal, symlinks and changed payloads.

    This validates a staging contract; it is not an OS sandbox. Run downstream
    workers under an account/container without access to raw or truth mounts.
    """
    if record.get('identity') != 'REVISION_REPLICATION':
        raise PermissionError('historical or unspecified identity')
    if record.get('split') not in ALLOWED.get(role, set()):
        raise PermissionError('split prohibited for role')
    root = Path(root).resolve(strict=True)
    path = (root / record['path']).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise PermissionError('artifact escapes staging root')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != record['sha256']:
        raise PermissionError('artifact hash mismatch')
    return path
