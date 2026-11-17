import asyncio, sys, types, tempfile, json, logging
from pathlib import Path
import stub
fi=types.ModuleType("toolboxv2.flows.isaa"); fi.__file__=stub.R+"/flows/isaa/__init__.py"
sys.modules["toolboxv2.flows"]=types.ModuleType("toolboxv2.flows"); sys.modules["toolboxv2.flows.isaa"]=fi
from toolboxv2.mods.isaa.extras.icli2.backend import IsaaBackend
tmp=Path(tempfile.mkdtemp()); (tmp/"Agents/self").mkdir(parents=True); (tmp/"Agents/self/agent.json").write_text('{"name":"self","fast_llm_model":"a"}')
(tmp/"Agents/myagent").mkdir(); (tmp/"Agents/myagent/agent.json").write_text("{}"); (tmp/"Agents/myagent/myagent.tar.gz").write_text("x")
class Mount:  local_path="/x"
class VFS:
    def __init__(s): s.mounts={}
    def mount(s,local_path,vfs_path): s.mounts[vfs_path]=Mount(); return {"success":True,"files_indexed":3}
    def unmount(s,at,save_changes=True): s.mounts.pop(at); return {"success":True,"files_saved":["a"]}
    def sync_all(s): return {"synced":["a","b"]}
class Sess:
    def get_history(s,n): return [{"role":"system","content":"sys"},{"role":"user","content":"frage  eins"},{"role":"assistant","content":[{"type":"text","text":"antwort"},{"type":"image_url","image_url":{}}]}][-n:]
    def __init__(s): s.vfs=VFS(); s.cleared=False
    def clear_history(s): s.cleared=True
class SM:
    def __init__(s): s.s={}
    async def get_or_create(s,n): return s.s.setdefault(n,Sess())
    def list_sessions(s): return [None, *s.s]          # wie der echte SessionManager: auch leere Namen
class Eng:
    def __init__(s): s._session_last_run={"default":"r1"}; s._active_executions={"r1":1}
class Agent:
    async def a_stream(s,query,session_id,max_iterations):
        s.asked=query
        yield {"type":"content","chunk":"Das ist ein ganzer Satz zum Vorlesen. "}
        yield {"type":"content","chunk":"unter","_sub_agent_id":"sub_1"}
        yield {"type":"final_answer","answer":"x"}
    bind_manager=types.SimpleNamespace(bindings={"coder":1})
    def __init__(s,n): s._execution_engine_cache=None; s.amd=types.SimpleNamespace(fast_llm_model="a",complex_llm_model="b"); s.session_manager=SM(); s.e=Eng()
    def _get_execution_engine(s): s._execution_engine_cache=s.e; return s.e
from toolboxv2.mods.isaa.extras.jobs import JobScheduler
ee=types.ModuleType("toolboxv2.mods.isaa.base.Agent.execution_engine")
class ExecutionContext:
    def __init__(s,run_id="",session_id="",status="paused"): s.run_id,s.session_id,s.status=run_id,session_id,status
    def to_checkpoint(s): return {"run_id":s.run_id,"session_id":s.session_id,"n":1}
    @classmethod
    def from_checkpoint(c,d): return c(d["run_id"],d["session_id"],"x")
ee.ExecutionContext=ExecutionContext
sys.modules["toolboxv2.mods.isaa.base.Agent"]=types.ModuleType("toolboxv2.mods.isaa.base.Agent"); sys.modules[ee.__name__]=ee
class Isaa:
    async def _default_cb(s,job): return {"dream":"report"}
    async def export_agent_network(s,names,path,entry_agent=None): Path(path).write_text("n"); s.exported=(names,entry_agent); return True,"ok"
    async def import_agent_network(s,path): return ({"netA":1,"netB":2},["w1"]) if Path(path).exists() else ({},["missing"])
    def __init__(s): s.job_scheduler=JobScheduler(tmp/"jobs.json", s._default_cb); s.config={"agents-name-list":["self","coder"],"agent-instance-self":1}; s.agent_data={"self":{}}; s.a={}
    async def get_agent(s,n): return s.a.setdefault(n,Agent(n))
    def agent_pack_path(s,n): return tmp/"Agents"/n/f"{n}.tar.gz"
isaa=Isaa()
app=types.SimpleNamespace(data_dir=str(tmp),appdata=str(tmp),get_mod=lambda n:isaa,logger=logging.getLogger("x"))
async def main():
    spawned=[]
    async def spawn(agent,session,query,kind="job",stream=None):
        out=""
        if stream is not None:
            async for c in stream: out=c.get("answer",out)
        spawned.append((agent,session,query,kind,out)); return out or "antwort"
    b=IsaaBackend(app); b.attach(spawn); r=await b.boot(); assert r["history"]==[] and r["mounts"]==0 and r["complex_model"]=="b", r
    rows=(await b.rows("agents"))["rows"]; print([ (x["label"],x["hint"]) for x in rows])
    assert {x["value"] for x in rows}=={"self","myagent","coder"} and "eigene Tools" in rows[0]["hint"]+rows[1]["hint"]
    m=(await b.rows("models"))["rows"]; print("models:",len(m), m[2]); assert len(m)>20 and m[0]["hint"]=="aktuell schnell"
    out=await b.do("model",name="openrouter/x",target="fast")
    assert out["ready"]["model"]=="openrouter/x" and json.loads((tmp/"Agents/self/agent.json").read_text())["fast_llm_model"]=="openrouter/x" and isaa.agent_data["self"]["fast_llm_model"]=="openrouter/x"
    d=tmp/"proj"; d.mkdir(); d2=tmp/"lib"; d2.mkdir()
    o=await b.do("mount",path=str(d)); assert o["ready"]["mounts"]==1 and "/project" in o["text"]
    o=await b.do("mount",path=str(d2)); assert "/lib" in o["text"]
    assert [x["value"] for x in (await b.rows("mounts"))["rows"]]==["/project","/lib","*"]
    assert "2 Dateien" in (await b.do("unmount",at="*"))["text"]; assert (await b.do("unmount",at="/lib"))["ready"]["mounts"]==1
    try: await b.do("mount",path="/gibt/es/nicht"); assert 0
    except FileNotFoundError as e: print("ok:",e)
    o=await b.do("session",name="+"); assert o["ready"]["session"].startswith("s-") and o["ready"]["mounts"]==0
    assert (await b.rows("sessions"))["rows"][0]["value"]=="+"
    await b.do("session",name="default"); await b.do("session_clear"); a=isaa.a["self"]; assert a.session_manager.s["default"].cleared and not a.e._active_executions
    try: await b.do("agent",name="nix"); assert 0
    except LookupError: pass
    assert (await b.do("agent",name="myagent"))["ready"]["agent"]=="myagent"
    await b.do("remember",text="hallo\nwelt"); await b.do("remember",text="/q"); assert b._read_history()==["hallo\nwelt","/q"]
    # ── #audio: nur diese Antwort wird gesprochen, Tag kommt nicht beim Agenten an ──
    said=[]; b._voice_begin=lambda: asyncio.sleep(0,True); b._voice_feed=said.append; b._voice_flush=lambda: said.append("<flush>")
    [c async for c in b.stream("myagent","default","frage ohne ton")]; assert said==[] and isaa.a["myagent"].asked=="frage ohne ton"
    [c async for c in b.stream("myagent","default","lies vor  #audio ")]; assert said==["Das ist ein ganzer Satz zum Vorlesen. ","<flush>"], said
    assert isaa.a["myagent"].asked=="lies vor"
    b._voice_begin=lambda: asyncio.sleep(0,False); said.clear()
    out=[c async for c in b.stream("myagent","default","x #audio")]; assert said==[] and "nicht verfügbar" in out[0]["status_msg"]
    del b._voice_begin, b._voice_feed, b._voice_flush
    # ── Schritt 5: Rueckblick ──
    rc=(await b.do("session",name="default"))["ready"]["recap"]; print("recap:",rc); assert rc==[("user","frage eins"),("assistant","antwort")]
    # ── Jobs gegen den echten JobScheduler ──
    sch=isaa.job_scheduler; assert sch._fire_callback==b._fire_job and (await b.rows("jobs"))["rows"]==[]
    for bad in ("nur text","x: y"):
        try: await b.do("job_add",spec=bad); assert 0
        except ValueError: pass
    print((await b.do("job_add",spec="alle 30m aufräumen: lösche alte logs"))["text"])
    await b.do("job_add",spec="um 9:05 bericht: schreibe: den bericht"); await b.do("dream"); assert "gibt es schon" in (await b.do("dream"))["text"]
    jobs=sch.list_jobs(); assert [j.name for j in jobs]==["aufräumen","bericht","auto-dream"], [j.name for j in jobs]
    assert jobs[0].trigger.interval_seconds==1800 and jobs[1].trigger.cron_expression=="5 9 * * *" and jobs[1].query=="schreibe: den bericht" and jobs[0].agent_name=="myagent"
    rows=(await b.rows("jobs"))["rows"]; print(rows[0]); assert len(rows)==3 and "on_interval · active" in rows[0]["hint"]
    jid=rows[0]["value"]; await b.do("job",job=jid,op="fire"); await asyncio.sleep(0.3)
    assert spawned[-1][:4]==("myagent","default","aufräumen: lösche alte logs","job"), spawned
    assert sch.get_job(jid).last_result=="completed" and sch.get_job(jid).run_count==1
    await b.do("job",job=rows[2]["value"],op="fire"); await asyncio.sleep(0.3); assert spawned[-1][4]=="{'dream': 'report'}", spawned[-1]
    await b.do("job",job=jid,op="pause"); assert sch.get_job(jid).status=="paused"; await b.do("job",job=jid,op="pause"); assert sch.get_job(jid).status=="active"
    await b.do("job",job=jid,op="remove"); assert len(sch.list_jobs())==2
    # ── Networks ──
    print((await b.do("network_save"))["text"]); assert isaa.exported==(["myagent","coder"],"myagent")
    ar=(await b.rows("agents"))["rows"]; assert ar[-1]["value"].startswith("net:") and "myagent" in ar[-1]["label"]
    o=await b.do("agent",name=ar[-1]["value"]); assert o["ready"]["agent"]=="netA" and "1 Warnungen" in o["text"]
    try: await b.do("network_load",path="gibtsnicht"); assert 0
    except FileNotFoundError: pass
    # ── Pausierte Laeufe: beim Beenden sichern, beim Start wiederherstellen ──
    a=isaa.a["self"]; a._get_execution_engine(); isaa.config["agent-instance-self"]=a
    a.e._active_executions={"p1":ExecutionContext("p1","work","paused"),"d1":ExecutionContext("d1","x","completed")}
    await b.close(); saved=json.loads((tmp/"icli2/resume.json").read_text()); assert list(saved)==["p1"] and saved["p1"]["agent_name"]=="self"
    assert sch._fire_callback==isaa._default_cb
    a.e._active_executions={}; a.e._session_last_run={}
    b2=IsaaBackend(app); b2.attach(spawn); await b2.boot()
    assert a.e._active_executions["p1"].status=="paused" and a.e._session_last_run=={"work":"p1"}; await b2.close()
    print("backend ok")
asyncio.run(main())
