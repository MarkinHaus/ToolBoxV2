import asyncio, tarfile, io, os, textwrap, tempfile, dill, types
from pathlib import Path
src=open(Path(__file__).resolve().parents[3]/"module.py").read()
a=src.index("    # ── Auto-Pack"); b=src.index("    async def load_all_agents")
tmp=Path(tempfile.mkdtemp())
class E:  # ToolEntry-Ersatz
    def __init__(s,name,fn,source="local"): s.name,s.function,s.source,s.description,s.category=name,fn,source,"d",["c"]
class TM:
    def __init__(s): s.r={}
    def get_all(s): return list(s.r.values())
    def exists(s,n): return n in s.r
class Agent:
    def __init__(s): s.tool_manager=TM()
    def add_tool(s,f,name=None,description=None,category=None): s.tool_manager.r[name]=E(name,f)
ns=dict(dill=dill,io=io,Path=Path,tarfile=tarfile,os=os,FlowAgent=Agent,get_app=lambda:types.SimpleNamespace(data_dir=str(tmp)),
        _get_serializer=lambda:dill,_deserialize_tool=lambda d,n:(dill.loads(d),None))
exec("class M:\n"+src[a:b]+'''
    def print(self,*a): print("   isaa:",*a)
    def _read_from_tar(self, tar, name, binary=False):
        try: return tar.extractfile(tar.getmember(name)).read()
        except KeyError: return None
    async def save_agent(self, name, path, include_checkpoint=True, custom_only=False):
        self.saved=getattr(self,"saved",0)+1
        data={e.name:{"data":dill.dumps(e.function),"description":e.description,"category":e.category}
              for e in self.agent.tool_manager.get_all() if not str(e.function.__module__).startswith("toolboxv2.")}
        Path(path).parent.mkdir(parents=True,exist_ok=True); raw=dill.dumps(data)
        with tarfile.open(path,"w:gz") as t:
            i=tarfile.TarInfo("tools.dill"); i.size=len(raw); t.addfile(i,io.BytesIO(raw))
''',ns)
M=ns["M"]
def my_tool(x:int)->int: return x*2
def sys_tool(): pass
sys_tool.__module__="toolboxv2.mods.isaa.x"
async def main():
    # Prozess A: Agent mit eigenem Tool wird gebaut -> Auto-Pack
    m=M(); a=Agent(); a.add_tool(my_tool,name="my_tool"); a.add_tool(sys_tool,name="sys_tool"); m.agent=a
    await m._auto_pack("myagent",a); assert m.agent_pack_path("myagent").exists() and m.saved==1
    # Prozess B: frisch aus agent.json gebaut, ohne eigenes Tool -> wird wieder angehaengt
    m2=M(); b=Agent(); b.add_tool(sys_tool,name="sys_tool"); m2.agent=b
    assert m2._restore_packed_tools("myagent",b)==["my_tool"] and b.tool_manager.r["my_tool"].function(21)==42
    # ohne Restore darf ein Agent ohne das Tool das Paket nicht ueberschreiben
    m3=M(); c=Agent(); c.add_tool(sys_tool,name="sys_tool"); c.add_tool(lambda:1,name="other"); m3.agent=c
    await m3._auto_pack("myagent",c); assert not hasattr(m3,"saved")
    # Agent ohne eigene Tools: kein Paket
    m4=M(); d=Agent(); d.add_tool(sys_tool,name="sys_tool"); m4.agent=d
    await m4._auto_pack("plain",d); assert not m4.agent_pack_path("plain").exists()
    print("auto-pack roundtrip ok")
asyncio.run(main())
