"""Python audit boundary for revision training. This is NOT an OS sandbox."""
import os
import sys
from pathlib import Path


def canonical(path):
    return os.path.normcase(str(Path(path).resolve()))


class TrainingAccess:
    def __init__(self, reads, writes, runtime_roots):
        self.reads = {canonical(p) for p in reads}
        self.writes = {canonical(p) for p in writes}
        self.runtime_roots = [Path(p).resolve() for p in runtime_roots]
        self.records = []
        self.phase = 'module_import'
        self.active = True
        self.busy = False

    def check(self, path, operation):
        if isinstance(path, int):
            self.records.append(dict(path=str(path), operation=operation,
                                     phase=self.phase, allowed=False, category='file_descriptor'))
            raise PermissionError('unregistered file descriptor')
        key = canonical(os.fsdecode(path))
        runtime = operation == 'read' and any(Path(key).is_relative_to(Path(os.path.normcase(str(r)))) for r in self.runtime_roots)
        allowed = key in (self.reads if operation == 'read' else self.writes) or runtime
        self.records.append(dict(path=key, operation=operation, phase=self.phase,
                                 allowed=allowed, category='runtime' if runtime else 'registered_input' if key in self.reads else 'output' if key in self.writes else 'prohibited'))
        if not allowed:
            raise PermissionError(f'{operation} not allowlisted: {key}')

    def hook(self, event, args):
        if not self.active or self.busy:
            return
        self.busy = True
        try:
            if event == 'open':
                path, mode, flags = args
                write = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
                self.check(path, 'write' if write else 'read')
                if flags & os.O_RDWR:
                    self.check(path, 'read')
            elif event in {'subprocess.Popen', 'os.system', 'os.exec', 'os.posix_spawn', 'ctypes.dlopen', 'ctypes.dlsym', 'mmap.__new__', 'os.link', 'os.symlink', 'os.rename', 'os.remove'}:
                self.records.append(dict(path=str(args[0]) if args else '', operation=event,
                                         phase=self.phase, allowed=False, category='blocked_capability'))
                raise PermissionError(f'capability disabled during training: {event}')
        finally:
            self.busy = False

    def install(self):
        sys.addaudithook(self.hook)
