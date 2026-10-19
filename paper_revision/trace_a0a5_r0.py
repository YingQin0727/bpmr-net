"""Non-elevated process-tree native I/O trace for the formal A0-A5 campaign."""
import json, sys, threading, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/revision_mtcomm_2026/REVISION_REPLICATION_training/DesignA'
TRACE_TOOLS=ROOT/'outputs/revision_mtcomm_2026/TRAINING_2D_ACCESS_RELEASE/trace_tools'
sys.path.insert(0,str(TRACE_TOOLS)); import frida
def main():
    args=sys.argv[1:]
    if args!=['--limit','720']: raise SystemExit('expected explicit --limit 720')
    errors=[]; sessions=[]; done=threading.Event(); lock=threading.Lock()
    source=(ROOT/'paper_revision/native_file_trace.js').read_text(encoding='utf-8')
    device=frida.get_local_device()
    dest=OUT/'REVISION_DESIGNA_A0A5_R0_NATIVE_ACCESS.jsonl'
    with dest.open('w',encoding='utf-8') as log:
        def write(r):
            with lock: log.write(json.dumps(r)+'\n'); log.flush()
        def instrument(pid,root=False):
            s=device.attach(pid); sessions.append(s); s.enable_child_gating()
            if root:s.on('detached',lambda *a:done.set())
            script=s.create_script(source)
            def message(msg,data):
                if msg['type']=='send':write(msg['payload'])
                else:
                    errors.append(str(msg));write({'instrumentation_error':msg,'pid':pid})
                    try:device.kill(pid)
                    except Exception:pass
            script.on('message',message); script.load(); s._revision_script=script
            write({'controller':'attached','pid':pid,'root':root})
        def child(c):
            try:instrument(c.pid);device.resume(c.pid)
            except Exception as e:
                errors.append(str(e));write({'controller_error':str(e),'pid':c.pid})
                try:device.kill(c.pid)
                except Exception:pass
        def on_child(c):threading.Thread(target=child,args=(c,),daemon=True).start()
        device.on('child-added',on_child)
        cfg=(ROOT/'.venv-revision311/pyvenv.cfg').read_text(encoding='utf-8')
        base=next(x.split('=',1)[1].strip() for x in cfg.splitlines() if x.startswith('home ='))
        site=ROOT/'.venv-revision311/Lib/site-packages'
        code="import site,sys,runpy;site.addsitedir("+repr(str(site))+ ");sys.argv=['controlled_a0a5_r0','--limit','720'];runpy.run_module('paper_revision.controlled_a0a5_r0',run_name='__main__')"
        pid=device.spawn([str(Path(base)/'python.exe'),'-B','-c',code],cwd=str(ROOT),stdio='inherit')
        instrument(pid,True);device.resume(pid)
        last=time.monotonic()
        while not done.wait(1):
            if time.monotonic()-last>60: print('A0-A5 native trace running',flush=True);last=time.monotonic()
    (OUT/'REVISION_DESIGNA_A0A5_R0_NATIVE_CONTROLLER.json').write_text(json.dumps({'frida_version':frida.__version__,'errors':errors,'root_pid':pid,'attached_processes':len(sessions),'trace':str(dest),'coverage':'User-mode ntdll file APIs with child gating; direct syscalls/pre-hook loader IO not exhaustively covered.'},indent=2),encoding='utf-8')
    if errors:raise SystemExit('native tracing error; campaign is not releasable')
if __name__=='__main__':main()
