import asyncio, sys, time
import stub
from toolboxv2.mods.isaa.extras.icli2 import protocol as P
from toolboxv2.mods.isaa.extras.icli2.engine import Engine
from toolboxv2.mods.isaa.extras.icli2.app import ICli2

class Fake:
    def __init__(s): s.active=set(); s.log=[]; s.spawn=None
    def attach(s,spawn): s.spawn=spawn
    async def boot(s):
        await asyncio.sleep(0.3); return dict(s.st, history=["alte eingabe"])
    async def stream(s, agent, session, query):
        s.active.add(session)
        try:
            yield {"type":"status","status_msg":"Preparing"}
            yield {"type":"reasoning","chunk":"hmm "}
            yield {"type":"tool_start","name":"vfs_shell","args":'{"cmd":"ls -la src", "timeout": 30}',"id":"t1"}
            if "block" in query: time.sleep(3.0)          # synchroner Tool-Call blockiert Engine-Loop
            yield {"type":"tool_result","name":"vfs_shell","id":"t1","result":'{"success": false, "stdout":"x"}'}
            yield {"type":"tool_start","name":"vfs_write","args":{"reason":"Konfiguration fuer den neuen Dienst anlegen","path":"/a/b.yaml","content":"a: 1\nb: 2","opts":{"mkdir":True,"mode":[1,2]}},"id":"t2"}
            yield {"type":"tool_result","name":"vfs_write","id":"t2","result":{"success":True,"size":2048,"lines":2,"file_type":"yaml"}}
            if "sub" in query:
                for sid,task in (("sub_a1","Recherche API"),("sub_b2","Tests schreiben")):
                    yield {"type":"tool_start","name":"vfs_read","args":{"path":"/x.py"},"id":"s"+sid,"_sub_agent_id":sid,"_sub_agent_task":task}
                    yield {"type":"content","chunk":f"Teilergebnis {sid}. ","_sub_agent_id":sid,"_sub_agent_task":task}
                    yield {"type":"tool_result","name":"vfs_read","id":"s"+sid,"result":{"success":True,"lines":3},"_sub_agent_id":sid,"_sub_agent_task":task}
                    yield {"type":"done","success":False,"_sub_agent_id":sid,"_sub_agent_task":task}
            for i in range(400 if "long" in query else 20):
                yield {"type":"content","chunk":f"wort{i} " + ("\n\n# Titel\n\n" if i==5 else "")}
                await asyncio.sleep(0.005 if "long" not in query else 0.02)
            yield {"type":"final_answer","answer":"ok"}
            yield {"type":"done","success":True}
        finally: s.active.discard(session)
    st={"agent":"self","session":"default","model":"fake-1","complex_model":"fake-c","mounts":0,"user":""}
    async def rows(s,what):
        R=lambda v,h="":{"value":v,"label":v,"hint":h}
        return {"agents":{"title":"Agent wechseln","rows":[R("self","geladen · aktiv"),R("myagent","gespeichert · eigene Tools"),R("coder","registriert")]},
                "models":{"title":"Modell wählen","rows":[R("openrouter/a"),R("gateway/big-model")]},
                "sessions":{"title":"Session wechseln","rows":[R("+"),R("default","aktiv"),R("work")]},
                "tb":{"title":"ToolBox-Funktionen","rows":[R("CloudM.get_user","name"),R("isaa.version")]},
                "shares":{"title":"LiveSync","rows":[R("sh1","getrennt")]},
                "voice":{"title":"Sprache","rows":[R("speak","aus"),R("omni","aus")]},
                "jobs":{"title":"Jobs","rows":[{"value":"j1","label":"nightly","hint":"on_cron · active · completed · 3x · self"}]},
                "mounts":{"title":"Mounts","rows":[R("/project","/tmp/x")] if s.st["mounts"] else []} }[what] | ({"title":"Keine Mounts – /mount <ordner>"} if what=="mounts" and not s.st["mounts"] else {})
    async def do(s,action,**a):
        s.log.append((action,a))
        if action=="agent":
            if a["name"]=="kaputt": raise LookupError("Agent 'kaputt' ist nicht registriert")
            s.st.update(agent=a["name"],model="fake-"+a["name"])
        elif action=="model": s.st.update(model=a["name"])
        elif action=="session":
            s.st.update(session="s-new" if a["name"]=="+" else a["name"]); return {"ready":dict(s.st,recap=[("user","alte frage"),("assistant","alte antwort")])}
        elif action=="job":
            asyncio.ensure_future(s.spawn("self","default","nightly: aufräumen",kind="job")); return {"text":"Job nightly gestartet"}
        elif action in ("job_add","dream","network_save","network_load"): return {"text":action+" ok"}
        elif action=="login": s.st.update(user="markin"); return {"ready":dict(s.st),"text":"Angemeldet als markin"}
        elif action=="tb": return {"text":"ergebnis\n  wert  "+a["target"]+" "+a.get("args","")}
        elif action=="share": return {"text":"share "+a["share"]+" verbunden"}
        elif action=="voice": return {"text":"Vorlesen an"}
        elif action=="voice_toggle":
            s.rec=not getattr(s,"rec",False); return {"text":"● Aufnahme läuft – F4 beendet"} if s.rec else {"draft":"gesprochener text"}
        elif action=="mount": s.st.update(mounts=1); return {"ready":dict(s.st),"text":"/tmp/x → /project · 12 Dateien"}
        elif action=="unmount": s.st.update(mounts=0); return {"ready":dict(s.st),"text":"/project ausgehängt"}
        elif action=="pack": return {"text":"myagent gepackt: 3 Tools gesichert"}
        elif action in ("remember","session_clear"): return {"text":"Session geleert"} if action=="session_clear" else {}
        return {"ready":dict(s.st)}
    async def is_running(s,a,se): return se in s.active
    async def stop(s,a,se): pass
    async def pause(s,a,se): return False
    def tool_meta(s,name,rd): return "2.0kb 2L yaml" if name=="vfs_write" else ""
    async def close(s): pass

async def main():
    eng=Engine(Fake()); eng.start()
    app=ICli2(eng)
    async with app.run_test(size=(90,28)) as pilot:
        await pilot.pause(0.6)
        assert app.ready and app.model=="fake-1"
        await pilot.press(*"hallo"); await pilot.press("enter")
        await pilot.pause(1.0)
        app.save_screenshot("/tmp/icli2_shot_start.svg")
        from textual.widgets import Markdown
        txt=[m.source for m in app.query(Markdown)]
        assert "wort19" in txt[-1], txt
        assert not app.active
        cards=list(app.query(".tool")); assert cards and "✗" in str(cards[0].title), cards[0].title
        for c in cards: print("  ", c.title); assert "{" not in str(c.title)
        assert "cmd using ls -la src" in str(cards[0].title) and "Konfiguration" in str(cards[1].title) and "path" not in str(cards[1].title)
        body=str(cards[1]._body.render()); print(body); assert "{" not in body and "[" not in body
        assert app.query_one("#row").has_class("-done")
        await pilot.press("alt+1"); await pilot.pause(0.1); assert [c.collapsed for c in cards]==[True,False]
        await pilot.press("ctrl+o"); await pilot.pause(0.1); assert [c.collapsed for c in cards]==[False,False]
        await pilot.press("ctrl+o"); await pilot.pause(0.1); assert [c.collapsed for c in cards]==[True,True]
        await pilot.press("ctrl+r"); await pilot.pause(0.1); assert not app.think_boxes[0].collapsed
        await pilot.press("ctrl+r"); await pilot.press("alt+2"); await pilot.pause(0.2)
        print("stream ok")
        # UI bleibt bedienbar waehrend Engine-Loop blockiert ist
        await pilot.press(*"block"); await pilot.press("enter"); await pilot.pause(0.4)
        t=time.monotonic(); worst=0
        for ch in "tippen geht":
            t0=time.monotonic(); await pilot.press("space" if ch==" " else ch); worst=max(worst,time.monotonic()-t0)
        await pilot.pause(0.8)
        foot=str(app.query_one("#foot").render())
        print("typed during block:", repr(app.prompt.text), f"worst key {worst*1000:.0f}ms", "| lag", round(eng.lag(),1), "|", foot)
        assert app.prompt.text=="tippen geht" and eng.lag()>1
        await pilot.pause(2.5)
        app.prompt.set_text("")
        # Stopp per Esc
        await pilot.press(*"long"); await pilot.press("enter"); await pilot.pause(0.5)
        assert app.active; await pilot.press("escape"); await pilot.pause(0.4)
        assert not app.active
        notes=[str(n.render()) for n in app.query(".note")]
        print("notes:", notes)
        assert any("gestoppt" in n for n in notes)
        # Agent-Picker: Liste live, fuzzy filtern, wechseln
        await pilot.press(*"/agent"); await pilot.press("enter"); await pilot.pause(0.3)
        from toolboxv2.mods.isaa.extras.icli2.picker import Picker
        assert isinstance(app.screen, Picker)
        app.save_screenshot("/tmp/icli2_shot_picker.svg")
        await pilot.press(*"mya"); await pilot.pause(0.1); await pilot.press("enter"); await pilot.pause(0.4)
        assert app.agent=="myagent" and app.model=="fake-myagent", (app.agent, app.model)
        await pilot.press(*"/agent"); await pilot.press("space"); await pilot.press(*"kaputt"); await pilot.press("enter"); await pilot.pause(0.3)
        assert app.agent=="myagent"
        await pilot.press(*"/pack"); await pilot.press("enter"); await pilot.pause(0.3)
        notes=[str(n.render()) for n in app.query(".note")]; print("notes2:", notes[-3:])
        assert any("nicht registriert" in n for n in notes) and any("gepackt" in n for n in notes)
        fake=eng._backend; P_=Picker
        # Modell: zwei Picker (Modell, Ziel)
        await pilot.press(*"/model"); await pilot.press("enter"); await pilot.pause(0.3); assert isinstance(app.screen,P_)
        await pilot.press(*"big"); await pilot.press("enter"); await pilot.pause(0.2); assert isinstance(app.screen,P_)
        app.save_screenshot("/tmp/icli2_shot_model.svg")
        await pilot.press("down"); await pilot.press("enter"); await pilot.pause(0.3)
        assert ("model",{"name":"gateway/big-model","target":"fast"}) in fake.log and app.model=="gateway/big-model", fake.log[-3:]
        # Session neu + direkt
        await pilot.press(*"/new"); await pilot.press("enter"); await pilot.pause(0.3); assert app.session=="s-new"
        await pilot.press(*"/session"); await pilot.press("space"); await pilot.press(*"work"); await pilot.press("enter"); await pilot.pause(0.3); assert app.session=="work"
        # Mount: ohne Mounts Hinweis, dann mounten, dann ueber Picker aushaengen
        await pilot.press(*"/mount"); await pilot.press("enter"); await pilot.pause(0.3); assert not isinstance(app.screen,P_)
        await pilot.press(*"/mount"); await pilot.press("space"); await pilot.press(*"/tmp/x"); await pilot.press("enter"); await pilot.pause(0.3); assert app.mounts==1
        await pilot.press(*"/mount"); await pilot.press("enter"); await pilot.pause(0.3); assert isinstance(app.screen,P_); await pilot.press("enter"); await pilot.pause(0.3); assert app.mounts==0
        # Palette per Strg+P -> /reset
        await pilot.press("ctrl+p"); await pilot.pause(0.2); assert isinstance(app.screen,P_); app.save_screenshot("/tmp/icli2_shot_palette.svg")
        await pilot.press(*"rese"); await pilot.press("enter"); await pilot.pause(0.3); assert ("session_clear",{}) in fake.log
        # History: Pfeil hoch holt letzte Eingabe, persistiert wurde ueber die Engine
        await pilot.press("up"); assert app.prompt.text=="/mount", app.prompt.text
        await pilot.press("down"); await pilot.press("down"); assert app.prompt.text==""
        assert ("remember",{"text":"/new"}) in fake.log and app.history[0]=="alte eingabe"
        print("step4 ui ok:", [n for n in [str(x.render()) for x in app.query(".note")][-6:]])
        # ── Schritt 5 ──
        # Rueckblick beim Session-Wechsel
        await pilot.press(*"/session"); await pilot.press("space"); await pilot.press(*"default"); await pilot.press("enter"); await pilot.pause(0.3)
        notes=[str(x.render()) for x in app.query(".note")]; assert any("alte antwort" in n for n in notes), notes[-4:]
        # mehrzeilige Eingabe: Strg+J = neue Zeile, Enter sendet
        await pilot.press(*"zeile1"); await pilot.press("ctrl+j"); await pilot.press(*"sub"); assert app.prompt.text=="zeile1\nsub"
        await pilot.press("enter"); await pilot.pause(1.0)
        assert app.prompt.text=="" and app.history[-1]=="zeile1\nsub"
        run=list(app.runs.values())[-1]; assert run.status=="done", run.status
        print("subs:", list(run.subs)); assert list(run.subs)==["Recherche API","Tests schreiben"]
        assert run.subs["Recherche API"].answer.startswith("Teilergebnis sub_a1") and "Teilergebnis" not in run.answer
        assert len(list(run.box.query(".sub")))==2 and len(list(run.subs["Tests schreiben"].box.query(".tool")))==1
        # zwei Laeufe parallel: getrennte Bloecke, Fokus wechseln, nur der fokussierte wird gestoppt
        await pilot.press(*"long eins"); await pilot.press("enter"); await pilot.pause(0.3)
        await pilot.press("/", *"new"); await pilot.press("enter"); await pilot.pause(0.3)
        await pilot.press(*"long zwei"); await pilot.press("enter"); await pilot.pause(0.5)
        act=app.active; assert len(act)==2, len(act)
        (r1,v1),(r2,v2)=act.items(); assert app.focus_run==r2 and v2.box.has_class("-focus") and v1.box is not v2.box
        assert "wort" in v1.answer and "wort" in v2.answer
        await pilot.press("f2"); await pilot.pause(2.6)
        side=str(app.query_one("#side-text").render()); print(side); assert "Läufe" in side and "nightly" in side and "long eins" in side
        app.save_screenshot("/tmp/icli2_shot_side.svg")
        await pilot.press("f3"); await pilot.pause(0.1); assert app.focus_run==r1
        await pilot.press("escape"); await pilot.pause(0.5); assert v1.status=="stopped" and v2.status=="running" and app.focus_run==r2
        await pilot.press("escape"); await pilot.pause(0.5); assert not app.active
        # /task Picker
        await pilot.press(*"/task"); await pilot.press("enter"); await pilot.pause(0.2); assert isinstance(app.screen,P_); await pilot.press("escape"); await pilot.pause(0.1)
        # Job ueber Picker ausfuehren -> von der Engine gestarteter Lauf erscheint in der UI
        await pilot.press(*"/job"); await pilot.press("enter"); await pilot.pause(0.3); assert isinstance(app.screen,P_)
        await pilot.press("enter"); await pilot.pause(0.2); assert isinstance(app.screen,P_); await pilot.press("enter"); await pilot.pause(1.2)
        jr=list(app.runs.values())[-1]; assert jr.query=="nightly: aufräumen" and jr.status=="done", (jr.query, jr.status)
        heads=[str(x.render()) for x in jr.box.query(".note")]; assert "job · self" in heads[0], heads
        await pilot.press(*"/dream"); await pilot.press("enter"); await pilot.pause(0.2); assert ("dream",{}) in fake.log
        await pilot.press(*"/job"); await pilot.press("space"); await pilot.press(*"alle 5m x: y"); await pilot.press("enter"); await pilot.pause(0.2); assert ("job_add",{"spec":"alle 5m x: y"}) in fake.log
        await pilot.press(*"/network"); await pilot.press("enter"); await pilot.pause(0.2); assert ("network_save",{}) in fake.log
        await pilot.press("f2")
        print("step5 ui ok")
        # ── Schritt 8/9 ──
        await pilot.press(*"/login"); await pilot.press("enter"); await pilot.pause(0.3)
        assert app.user=="markin" and "markin" in str(app.query_one("#head").render())
        await pilot.press(*"/tb"); await pilot.press("enter"); await pilot.pause(0.3); assert isinstance(app.screen,P_)
        await pilot.press(*"vers"); await pilot.press("enter"); await pilot.pause(0.3); assert ("tb",{"target":"isaa.version"}) in fake.log
        await pilot.press(*"/tb"); await pilot.press("space"); await pilot.press(*"CloudM.get_user"); await pilot.press("space"); await pilot.press(*"a=1"); await pilot.press("enter"); await pilot.pause(0.3)
        assert ("tb",{"target":"CloudM.get_user","args":"a=1"}) in fake.log
        await pilot.press(*"/share"); await pilot.press("enter"); await pilot.pause(0.3); await pilot.press("enter"); await pilot.pause(0.3); assert ("share",{"share":"sh1"}) in fake.log
        await pilot.press(*"/voice"); await pilot.press("enter"); await pilot.pause(0.3); await pilot.press("enter"); await pilot.pause(0.3); assert ("voice",{"op":"speak"}) in fake.log
        await pilot.press("f4"); await pilot.pause(0.3); await pilot.press(*"vorher"); await pilot.press("f4"); await pilot.pause(0.3)
        assert app.prompt.text=="vorher gesprochener text", app.prompt.text
        app.prompt.set_text("")
        print("step8/9 ui ok")
        await pilot.press(*"logs"); await pilot.press("enter"); await pilot.pause(0.8)
        app.save_screenshot("/tmp/icli2_shot.svg")
        await pilot.press(*"/q"); await pilot.press("enter")
    eng.send(P.SHUTDOWN); eng.join(5)
    assert not eng._thread.is_alive(); print("clean shutdown")
asyncio.run(main())
