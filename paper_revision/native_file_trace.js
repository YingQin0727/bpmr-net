// User-mode Windows native I/O observation. Not an OS security boundary.
const nt = Process.getModuleByName('ntdll.dll');
let phase = 'process_bootstrap';
let seq = 0;
const handles = new Map();
function forbidden(path) {
  const p=path.toLowerCase();
  return p.includes('\\test_truth\\') || p.includes('\\official_download\\') ||
         p.includes('\\$recycle.bin\\') || p.endsWith('.raw');
}
function deniedObject() {
  // Redirect a denied open to an invalid object namespace before the syscall.
  // Never pass the prohibited scientific path into the original Nt* function.
  const text='\\RevisionDeniedNamespace\\NON_SCIENTIFIC_TEST';
  const name=Memory.allocUtf16String(text);
  const us=Memory.alloc(16);us.writeU16(text.length*2);us.add(2).writeU16(text.length*2+2);us.add(8).writePointer(name);
  const oa=Memory.alloc(48);oa.writeU32(48);oa.add(8).writePointer(ptr(0));oa.add(16).writePointer(us);
  oa.add(24).writeU32(0);oa.add(32).writePointer(ptr(0));oa.add(40).writePointer(ptr(0));
  return {oa:oa,us:us,name:name};
}
const query = new NativeFunction(nt.getExportByName('NtQueryObject'),'int',['pointer','uint','pointer','uint','pointer']);
const fileType = new NativeFunction(Process.getModuleByName('kernelbase.dll').getExportByName('GetFileType'),'uint',['pointer']);
let querying = false;
function handleName(handle) {
  const known=handles.get(handle.toString()); if(known) return known;
  if(querying) return '<unresolved-handle>';
  querying=true;
  try {
    const buf=Memory.alloc(65536), size=Memory.alloc(4);
    if(query(handle,1,buf,65536,size)>=0 && buf.readU16()>0) {
      const name=buf.add(8).readPointer().readUtf16String(buf.readU16()/2);
      handles.set(handle.toString(),name);return name;
    }
    const type=fileType(handle);
    if(type===3) return '\\Device\\AnonymousPipe';
    if(type===2) return '\\Device\\CharacterDevice';
  } catch(_) {} finally {querying=false;}
  return '<unresolved-handle>';
}
const wide = p => p.isNull() ? '' : p.readUtf16String();
function objectName(p) {
  try {
    const root = p.add(8).readPointer();
    const us = p.add(16).readPointer();
    const name = us.add(8).readPointer().readUtf16String(us.readU16() / 2);
    return (root.isNull() ? '' : handleName(root) + '\\') + name;
  } catch (_) { return '<unresolved-object>'; }
}
function emit(o) { send(Object.assign({pid:Process.id, phase:phase, seq:seq++}, o)); }
for (const name of ['NtCreateFile', 'NtOpenFile']) {
  Interceptor.attach(nt.getExportByName(name), {
    onEnter(args) {
      this.out = args[0]; this.path = objectName(args[2]);
      this.access = args[1].toUInt32();
      this.options=(name==='NtCreateFile'?args[8]:args[5]).toUInt32();
      this.disposition=name==='NtCreateFile'?args[7].toUInt32():1;
      this.blocked=forbidden(this.path);
      if(this.blocked) {this.denied=deniedObject();args[2]=this.denied.oa;}
    },
    onLeave(ret) {
      let handle = '';
      const originalStatus=ret.toInt32();
      if(this.blocked) ret.replace(ptr('0xc0000022'));
      if (ret.toInt32() >= 0) {
        handle = this.out.readPointer().toString(); handles.set(handle, this.path);
      }
      emit({api:name,path:this.path,access:this.access,status:ret.toInt32(),handle:handle,
            blocked:this.blocked,original_status:originalStatus,options:this.options,disposition:this.disposition});
    }
  });
}
for (const name of ['NtReadFile', 'NtWriteFile', 'NtCreateSection', 'NtMapViewOfSection']) {
  Interceptor.attach(nt.getExportByName(name), {
    onEnter(args) {
      this.handle = (name === 'NtCreateSection' ? args[6] : args[0]).toString();
      this.path = handleName(ptr(this.handle));
      if (name === 'NtCreateSection') this.out = args[0];
      this.length = (name === 'NtReadFile' || name === 'NtWriteFile') ? args[6].toUInt32() : 0;
    },
    onLeave(ret) {
      if (name === 'NtCreateSection' && ret.toInt32() >= 0) handles.set(this.out.readPointer().toString(),this.path);
      emit({api:name,path:this.path,handle:this.handle,status:ret.toInt32(),requested_bytes:this.length});
    }
  });
}
Interceptor.attach(nt.getExportByName('NtClose'), {
  onEnter(args) { this.handle = args[0].toString(); },
  onLeave(ret) { if (ret.toInt32() >= 0) handles.delete(this.handle); }
});
// Phase markers travel through OutputDebugStringW, never a data file.
Interceptor.attach(Process.getModuleByName('kernelbase.dll').getExportByName('OutputDebugStringW'), {
  onEnter(args) { const s=wide(args[0]); if(s.startsWith('REVISION_PHASE:')) {phase=s.slice(15);emit({api:'phase',path:phase});} }
});
emit({api:'trace_ready',path:'NtCreateFile/NtOpenFile/NtReadFile/NtWriteFile/NtCreateSection/NtMapViewOfSection'});
