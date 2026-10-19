"""Exact atomic commit capabilities layered over the frozen Python read guard.
Not an OS sandbox: native/direct-syscall coverage remains separately documented.
"""
import os
from pathlib import Path
from paper_revision.training_access import TrainingAccess, canonical

class AtomicTrainingAccess(TrainingAccess):
    def __init__(self, reads, writes, runtime_roots, *, namespace, pairs, immutable=()):
        super().__init__(reads, writes, runtime_roots)
        self.namespace = Path(namespace).absolute()
        self.pairs = {(canonical(a), canonical(b)) for a,b in pairs}
        self.immutable = {canonical(p) for p in immutable}

    def safe_path(self, value):
        p = Path(os.fsdecode(value))
        if not p.is_absolute() or '..' in p.parts:
            raise PermissionError('atomic paths must be absolute without traversal')
        for part in (p, *p.parents):
            if part.exists() or part.is_symlink():
                st = part.lstat()
                if part.is_symlink() or getattr(st, 'st_file_attributes', 0) & 0x400:
                    raise PermissionError('link/reparse point prohibited')
        key = canonical(p)
        if not Path(key).is_relative_to(Path(canonical(self.namespace))):
            raise PermissionError('outside atomic output namespace')
        return key

    def check(self, path, operation):
        if operation == 'write':
            key = self.safe_path(path)
            if key in self.immutable and Path(key).exists():
                raise PermissionError('frozen checkpoint cannot be overwritten')
        return super().check(path, operation)

    def hook(self, event, args):
        if event != 'os.rename' or not self.active or self.busy:
            return super().hook(event, args)
        self.busy = True
        record = dict(operation=event, source=str(args[0]), target=str(args[1]), phase=self.phase, allowed=False)
        try:
            if any(x not in (None, -1) for x in args[2:]):
                raise PermissionError('dir_fd atomic operations prohibited')
            source, target = self.safe_path(args[0]), self.safe_path(args[1])
            if (source,target) not in self.pairs or source not in self.writes or target not in self.writes:
                raise PermissionError('unregistered atomic source/target pair')
            if target in self.immutable and Path(target).exists():
                raise PermissionError('frozen checkpoint cannot be replaced')
            record.update(source=source,target=target,allowed=True)
        finally:
            self.records.append(record)
            self.busy = False
