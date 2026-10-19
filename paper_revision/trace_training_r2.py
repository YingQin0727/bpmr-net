"""Non-elevated Frida process-tree trace of the actual R2 verification launcher.

Frida is installed only in the ignored local trace_tools directory. Run this
controller with the Python version matching that private installation (D:/python).
"""
import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/revision_mtcomm_2026/TRAINING_2D_ACCESS_RELEASE'
sys.path.insert(0,str(OUT/'trace_tools'))
import frida


def main():
    formal = sys.argv[1:] == ['--source-b0']
    if sys.argv[1:] and not formal:raise SystemExit('Only --source-b0 is supported')
    destination = OUT/('source_b0_native.jsonl' if formal else 'r2_native.jsonl')
    errors = []
    sessions = []
    done = threading.Event()
    lock = threading.Lock()
    source = (ROOT/'paper_revision/native_file_trace.js').read_text(encoding='utf-8')
    device = frida.get_local_device()
    with destination.open('w',encoding='utf-8') as log:
        def write(record):
            with lock:
                log.write(json.dumps(record)+'\n')
                if record.get('api')=='phase' or 'controller' in record:log.flush()
        def instrument(pid, root=False):
            session = device.attach(pid)
            sessions.append(session)
            session.enable_child_gating()
            if root:
                session.on('detached',lambda *args: done.set())
            script = session.create_script(source)
            def message(msg,data):
                if msg['type']=='send': write(msg['payload'])
                else:
                    errors.append(msg);write(dict(instrumentation_error=msg,pid=pid))
                    try:device.kill(pid)
                    except Exception:pass
            script.on('message',message)
            script.load()
            # Keep a Python reference to the script for the entire process lifetime.
            session._revision_script = script
            write(dict(controller='attached',pid=pid,root=root))
        def attach_child(child):
            try:
                write(dict(controller='child_added',pid=child.pid,parent_pid=child.parent_pid))
                instrument(child.pid)
                device.resume(child.pid)
            except Exception as exc:
                errors.append(str(exc));write(dict(controller_error=str(exc)))
                try: device.kill(child.pid)
                except Exception: pass
        def child_added(child):
            # Frida callbacks must not synchronously wait on its dispatcher.
            threading.Thread(target=attach_child,args=(child,),daemon=True).start()
        device.on('child-added',child_added)
        # Windows venv python.exe is a redirector which creates another process.
        # Spawn the exact underlying interpreter directly with the same site-packages.
        config=(ROOT/'.venv-revision311/pyvenv.cfg').read_text()
        base=next(line.split('=',1)[1].strip() for line in config.splitlines() if line.startswith('home ='))
        entry="from paper_revision.training_campaign_2d import train_source_b0;train_source_b0()" if formal else "runpy.run_module('paper_revision.training_release_r2',run_name='__main__')"
        bootstrap="import site,runpy;site.addsitedir("+repr(str(ROOT/'.venv-revision311/Lib/site-packages'))+");"+entry
        args = [str(Path(base)/'python.exe'),'-B','-c',bootstrap]
        pid = device.spawn(args,cwd=str(ROOT),stdio='inherit')
        instrument(pid,root=True)
        device.resume(pid)
        last = time.monotonic()
        while not done.wait(1):
            if time.monotonic()-last > 45:
                print('R2 native trace running',flush=True);last=time.monotonic()
    (OUT/('source_b0_trace_controller.json' if formal else 'r2_trace_controller.json')).write_text(json.dumps(dict(frida_version=frida.__version__,errors=errors,
        root_pid=pid,attached_processes=len(sessions),trace=str(destination),
        coverage='User-mode ntdll function interception from suspended spawn; child gating enabled. Pre-attach loader activity, direct syscalls and inherited pre-hook handles not exhaustively covered.'),indent=2),encoding='utf-8')
    if errors: raise SystemExit('Native trace reported errors; release must remain NO')


if __name__ == '__main__': main()
