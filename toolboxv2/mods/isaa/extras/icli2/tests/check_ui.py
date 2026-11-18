import asyncio, sys, time
import stub
OR_429 = ('litellm.RateLimitError: RateLimitError: OpenrouterException - {"error":{"message":"Rate limit '
          'exceeded: free-models-per-day","code":429}}')
SUB_TASK_A = "Recherchiere die API von Dienst X\nund liste alle Endpunkte mit Methode und Zweck auf.\nNutze nur die offizielle Doku."
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
            if query.startswith("fehler"):                 # Abbruch durch den Anbieter (wie litellm/Engine)
                yield {"type":"tool_start","name":"vfs_shell","args":{"command":"ls"},"id":"e1","iter":1,"max_iter":30}
                yield {"type":"tool_result","name":"vfs_shell","id":"e1","result":{"success":True}}
                yield {"type":"error","error":OR_429}
                yield {"type":"done","success":False,"final_answer":"Es ist ein Fehler aufgetreten"}
                return
            if query.startswith("limit"):
                for i in range(1, 6): yield {"type":"status","status_msg":f"Schritt {i}","iter":i,"max_iter":5}
                yield {"type":"max_iterations","answer":"Ich konnte die Aufgabe nicht abschliessen. Zusammenfassung: drei Dateien gelesen.","iter":5,"max_iter":5}
                yield {"type":"done","success":False,"final_answer":"…"}
                return
            if query.startswith("ohnestart"):              # wie im echten Lauf: tool_result ohne tool_start
                yield {"type":"tool_result","name":"vfs_shell","id":"x9","result":{"success":True,"stdout":"a"}}
                yield {"type":"max_iterations","answer":"Zusammenfassung","iter":1,"max_iter":1}
                yield {"type":"done","success":False,"final_answer":"Zusammenfassung"}
                return
            if query.startswith("absturz"):
                yield {"type":"warning","message":"Tool rejected — correction injected."}
                yield {"type":"done","success":False,"final_answer":"Agent crashed: KeyError: 'tool_calls'"}
                return
            yield {"type":"status","status_msg":"Preparing"}
            yield {"type":"reasoning","chunk":"hmm "}
            yield {"type":"tool_start","name":"vfs_shell","args":'{"cmd":"ls -la src", "timeout": 30}',"id":"t1"}
            if "block" in query: time.sleep(3.0)          # synchroner Tool-Call blockiert Engine-Loop
            yield {"type":"tool_result","name":"vfs_shell","id":"t1","result":'{"success": false, "stdout":"x"}'}
            yield {"type":"tool_start","name":"vfs_write","args":{"reason":"Konfiguration fuer den neuen Dienst anlegen","path":"/a/b.yaml","content":"a: 1\nb: 2","opts":{"mkdir":True,"mode":[1,2]}},"id":"t2"}
            yield {"type":"tool_result","name":"vfs_write","id":"t2","result":{"success":True,"size":2048,"lines":2,"file_type":"yaml"}}
            if "sub" in query:
                for sid,task,n in (("sub_a1",SUB_TASK_A,2),("sub_b2","Tests schreiben fuer das Modul x",30)):
                    meta={"task":task,"status":"running","iter":3,"max_iter":120,"tokens":12345,"budget":512000,
                          "timeout":1800,"started":time.time()-42,"error":""}
                    tag={"_sub_agent_id":sid,"_sub_agent_task":task[:60],"_sub_agent_meta":meta}
                    for k in range(n):
                        yield {"type":"tool_start","name":"vfs_read","args":{"path":f"/x{k}.py"},"id":f"s{sid}{k}",**tag}
                        yield {"type":"tool_result","name":"vfs_read","id":f"s{sid}{k}","result":{"success":True,"lines":3},**tag}
                    yield {"type":"content","chunk":f"Teilergebnis {sid}. ",**tag}
                    final=dict(meta, final=True, status="completed" if sid=="sub_a1" else "failed",
                               result="## Ergebnis A\n\nDie API hat **drei** Endpunkte." if sid=="sub_a1" else "",
                               error="" if sid=="sub_a1" else OR_429)
                    yield {"type":"status","status_msg":"Sub-Agent fertig","_sub_agent_id":sid,"_sub_agent_task":task[:60],"_sub_agent_meta":final}
            for i in range(400 if "long" in query else 20):
                yield {"type":"content","chunk":f"wort{i} " + ("\n\n# Titel\n\n" if i==5 else "")}
                await asyncio.sleep(0.005 if "long" not in query else 0.02)
            yield {"type":"final_answer","answer":"ok"}
            yield {"type":"done","success":True}
        finally: s.active.discard(session)
    st={"agent":"self","session":"default","model":"fake-1","complex_model":"fake-c","mounts":0,"user":"","features":[]}
    async def rows(s,what,arg=None):
        R=lambda v,h="":{"value":v,"label":v,"hint":h}
        if what=="vfs":
            tree=["/project/","/project/src/","/project/src/main.py","/project/README.md","/notes.txt"]
            base=arg.rsplit("/",1)[0]+"/" if "/" in arg else "/"
            kids=[p for p in tree if p.startswith(arg or "/") and p.rstrip("/").rsplit("/",1)[0]+"/"==base and p!=base]
            return {"title":"VFS "+base,"rows":[{"value":p,"label":p.rstrip("/").rsplit("/",1)[1]+("/" if p.endswith("/") else ""),"hint":"Ordner" if p.endswith("/") else "9 Zeichen"} for p in kids]}
        if what=="context": return {"title":"Kontext","rows":[R(str(i)) | {"label":l} for i,l in enumerate(["━━━━──────────────── 1.8 %","4.6k von 262k · fake","","Aufteilung","System Prompt Total   2422 ━━━━━━━"])]}
        if what=="settings": return {"title":"Einstellungen","rows":[{"value":"set:max_iterations","label":"Max. Iterationen","hint":"30"},{"value":"context","label":"Kontext","hint":""},{"value":"login","label":"Anmeldung","hint":""}]}
        return {"agents":{"title":"Agent wechseln","rows":[R("self","geladen · aktiv"),R("myagent","gespeichert · eigene Tools"),R("coder","registriert")]},
                "models":{"title":"Modell wählen","rows":[R("openrouter/a"),R("gateway/big-model")]},
                "sessions":{"title":"Session wechseln","rows":[R("+"),R("default","aktiv"),R("work")]},
                "features":{"title":"Features","rows":[R("coder","aus"),R("execute","aus")]},
                "skills":{"title":"Skills","rows":[R("sk1","trigger")]},
                "mcp":{"title":"MCP-Server","rows":[R("fs","npx …")]},
                "plugins":{"title":"Plugins","rows":[R("demo","2 Skills")]},
                "market":{"title":"Marketplace","rows":[R("owner/tool","offiziell · Werkzeug"),R("","nicht erreichbar")]},
                "markets":{"title":"Marketplaces","rows":[R("owner/market")]},
                "discord":{"title":"Discord","rows":[R("status","Verbindungsstatus"),R("connect","Bot starten")]},
                "telegram":{"title":"Telegram","rows":[R("status","Verbindungsstatus")]},
                "tb":{"title":"ToolBox-Funktionen","rows":[R("CloudM.get_user","name"),R("isaa.version")]},
                "shares":{"title":"LiveSync","rows":[R("sh1","getrennt")]},
                "voice":{"title":"Sprache","rows":[R("omni","aus")]},
                "runs":{"title":"Läufe der Agenten","rows":[R("self|r1","Schritt 4 · tests") | {"label":"self · default · ⚠ hängt"}]},
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
        elif action in ("job_add","dream","network_save","network_load","run"): return {"text":action+" ok"}
        elif action=="feature": s.st["features"]=["coder"]; return {"ready":dict(s.st),"text":"Feature coder an"}
        elif action in ("skill","mcp","plugin","mcp_add","discord","telegram","plugin_install","plugin_confirm","plugin_discard","plugin_remove","market_add","market_remove"): return {"text":action+" ok"}
        elif action=="vfs": return {"text":"INHALT "+a["path"]}
        elif action=="set": return {"text":"Max. Iterationen: "+a["value"]}
        elif action=="login": s.st.update(user="markin"); return {"ready":dict(s.st),"text":"Angemeldet als markin"}
        elif action=="tb": return {"text":"ergebnis\n  wert  "+a["target"]+" "+a.get("args","")}
        elif action=="share": return {"text":"share "+a["share"]+" verbunden"}
        elif action=="voice": return {"text":"Omni-Session aktiv"}
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
        txt=[m.source for m in app.query('.answer')]
        assert "wort19" in txt[-1], txt
        assert not app.active
        cards=list(app.query(".tool")); assert cards and "✗" in str(cards[0].title), cards[0].title
        for c in cards: print("  ", c.title); assert "{" not in str(c.title)
        assert "cmd using ls -la src" in str(cards[0].title) and "Konfiguration" in str(cards[1].title) and "path" not in str(cards[1].title)
        body=cards[1].body; print(body); assert "{" not in body and "[" not in body
        assert app.query_one("#row").has_class("-done")
        await pilot.press("alt+1"); await pilot.pause(0.1); assert [c.collapsed for c in cards]==[True,False]
        await pilot.press("ctrl+o"); await pilot.pause(0.1); assert [c.collapsed for c in cards]==[False,False]
        await pilot.press("ctrl+o"); await pilot.pause(0.1); assert [c.collapsed for c in cards]==[True,True]
        await pilot.press("ctrl+r"); await pilot.pause(0.1); assert not app.think_boxes[0].collapsed
        await pilot.press("ctrl+r"); await pilot.press("alt+2"); await pilot.pause(0.2)
        gap=app.query_one("#gap"); assert gap.size.height==1 and gap.region.y+1==app.query_one("#bottom").region.y+1
        assert app.chat.region.bottom<=gap.region.y, (app.chat.region, gap.region)            # Abstand direkt ueber der Eingabe
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
        print("subs:", list(run.subs)); assert list(run.subs)==["sub_a1","sub_b2"]
        a, b = run.subs["sub_a1"], run.subs["sub_b2"]
        assert a.task==SUB_TASK_A and a.label=="Recherchiere die API von Dienst X" and a.answer.startswith("Teilergebnis sub_a1")
        assert "Teilergebnis" not in run.answer and len(list(run.box.query(".sub")))==2
        assert a.status=="completed" and b.status=="failed" and b.error==OR_429
        comp=str(b.tile.compact.render()); print(comp)
        assert "Schritt 3/120" in comp and "12.3k/512.0k Tokens" in comp and "Klick" in comp and "28 weitere" in comp
        assert comp.count("vfs_read")==2 and "Anbieter-Limit erreicht (429)" in comp      # letzte 3 Aktionen: 2 Tools + Text
        assert "noch" not in comp                                                         # beendet: keine Restzeit
        assert len(list(b.tile.query(".tool")))==0                                        # klein: nur EIN Widget
        # gross: volle Aufgabe oben, alle Aktionen scrollbar, unten Ergebnis bzw. Fehler
        a.tile.scroll_visible(animate=False); await pilot.pause(0.2)
        await pilot.click(a.tile); await pilot.pause(0.3)
        assert a.tile.expanded and SUB_TASK_A.splitlines()[1] in str(a.tile.head.render())
        assert [str(w.render()) for w in a.tile.foot.query(".sub-label")]==["Ergebnis"]
        assert "drei" in a.tile.foot.query("FrozenMarkdown").first().source
        b.tile.scroll_visible(animate=False); await pilot.pause(0.2)
        await pilot.click(b.tile); await pilot.pause(0.4)
        acts=list(b.tile.list.query("FoldLine")); assert len(acts)==31, len(acts)       # 30 Tools + Text
        assert all(str(w.render()).strip() for w in acts), "jede Aktion zeigt ihren Titel (nicht leer)"
        assert "vfs_read" in str(acts[0].render()) and "Teilergebnis" in str(acts[-1].render())
        assert b.tile.list.max_scroll_y>0                                                  # eigener Scrollbereich
        prob=b.tile.foot.query(".problem").first(); assert "Anbieter-Limit erreicht (429)" in prob.title
        prob.collapsed=False; await pilot.pause(0.1); assert "Meldung von Openrouter" in str(prob.render())
        # echtes Click-Ereignis an die Aktion; Textual reicht es selbst nach oben weiter (bubble)
        from textual import events
        def click(w): w.post_message(events.Click(w, 0, 0, 0, 0, 1, False, False, False))
        acts[-2].collapsed=True; click(acts[-2]); await pilot.pause(0.3)
        assert not acts[-2].collapsed and b.tile.expanded                                  # Aktion auf, Kachel bleibt offen
        click(b.tile.head); await pilot.pause(0.3); assert not b.tile.expanded and b.tile.compact.is_attached
        app.save_screenshot("/tmp/icli2_shot_subtile.svg")
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
        # Loeschen nur nach Rueckfrage: Enter (Standard "nein") loescht nichts, erst "ja"
        for confirm, sent in ((["enter"], False), (["down", "enter"], True)):
            await pilot.press(*"/job"); await pilot.press("enter"); await pilot.pause(0.3)
            await pilot.press("enter"); await pilot.pause(0.2); await pilot.press("down", "down", "enter"); await pilot.pause(0.2)
            assert isinstance(app.screen,P_) and "wirklich löschen" in app.screen._title, "keine Rückfrage"
            await pilot.press(*confirm); await pilot.pause(0.3); assert not isinstance(app.screen,P_)
            assert (("job",{"job":"j1","op":"remove"}) in fake.log) is sent, fake.log[-3:]
        await pilot.press(*"/dream"); await pilot.press("enter"); await pilot.pause(0.2); assert ("dream",{}) in fake.log
        await pilot.press(*"/job"); await pilot.press("space"); await pilot.press(*"alle 5m x: y"); await pilot.press("enter"); await pilot.pause(0.2); assert ("job_add",{"spec":"alle 5m x: y"}) in fake.log
        await pilot.press(*"/network"); await pilot.press("enter"); await pilot.pause(0.2); assert ("network_save",{}) in fake.log
        await pilot.press("f2")
        print("step5 ui ok")
        # ── Fehler: klickbare Zeile mit verstaendlicher Meldung ──
        async def lauf(q):
            await pilot.press(*[("space" if c==" " else c) for c in q]); await pilot.press("enter")
            t=time.monotonic()
            while time.monotonic()-t<10:
                await pilot.pause(0.1)
                r=[v for v in app.runs.values() if v.query==q]
                if r and r[0].status!="running": return r[0]
            raise AssertionError(q)
        r=await lauf("fehler bitte"); assert r.status=="failed" and r.problem_line is not None
        line=r.problem_line; print("FEHLER", line.title)
        assert line.title.startswith("✗ Anbieter-Limit erreicht (429) – Rate limit exceeded") and line.collapsed
        line.scroll_visible(animate=False); await pilot.pause(0.2)
        await pilot.click(line); await pilot.pause(0.2); body=str(line.render())
        assert "Meldung von Openrouter" in body and "/model" in body and '"code": 429' in body
        assert app.w_row.has_class("-fail")
        r=await lauf("limit bitte"); assert r.status=="limit" and r.problem_line.title=="◷ Iterationslimit erreicht (5/5)"
        assert "Zusammenfassung: drei Dateien" in r.answer and "/set max_iterations" in r.problem_line.body
        assert app.w_row.has_class("-ask")
        r=await lauf("absturz bitte"); assert r.problem_line.title.startswith("✗ Interner Fehler im Agenten")
        assert any("Tool rejected" in str(n.render()) for n in r.box.query(".note"))     # Warnung sichtbar
        r=await lauf("ohnestart bitte"); cards=[w for w in r.box.children if w.has_class("tool")]
        assert len(cards)==1 and "vfs_shell" in cards[0].title and cards[0].title.startswith("✓"), [c.title for c in cards]
        assert r.status=="limit" and r.problem_line.title=="◷ Iterationslimit erreicht (1/1)"
        print("errors ui ok")
        # ── Reihenfolge trotz gesammelter Textausgabe: Gedanken, Tool, Text ──
        first=[r for r in app.runs.values() if r.query=="hallo"][0]
        order=[("think" if w.has_class("think") else "tool" if w.has_class("tool") else "answer" if w.has_class("answer") else "")
               for w in first.box.children]
        assert [o for o in order if o][:4]==["think","tool","tool","answer"], order
        from textual.widgets import Markdown as _MD
        assert all(isinstance(w, _MD) for w in first.box.children if w.has_class("answer")), "fertig = Markdown bleibt"
        # ── Nachsteuerung: Frage + Antwort IN den laufenden Lauf, Weiteres darunter ──
        await pilot.press(*"long basis"); await pilot.press("enter"); await pilot.pause(0.6)
        main=[v for v in app.active.values() if v.query=="long basis"][0]
        await pilot.press(*"kurze frage"); await pilot.press("enter"); await pilot.pause(1.2)
        steer=[v for v in app.runs.values() if v.query=="kurze frage"][0]
        assert steer.box.parent is main.box and steer.box.has_class("steer"), "Steering-Block im Hauptlauf"
        assert steer.status=="done" and main.status=="running" and app.focus_run in app.runs and app.runs[app.focus_run] is main
        heads=[str(w.children[0].render()) for w in app.chat.children if w.has_class("run") and w.children]
        assert not [h for h in heads if "kurze frage" in h], "kein eigener Block unten"
        idx=list(main.box.children).index(steer.box)
        await pilot.pause(0.6)
        later=[w for w in list(main.box.children)[idx+1:] if w.has_class("answer")]
        assert later and "wort" in later[0].source, "Hauptlauf schreibt unter der Nachsteuerung weiter"
        before=[w for w in list(main.box.children)[:idx] if w.has_class("answer")]
        assert before and all(isinstance(w, _MD) for w in before), "Abschnitt davor bleibt Markdown"
        await pilot.press("escape"); await pilot.press("escape"); await pilot.pause(0.6); assert not app.active
        # ── alte Bloecke werden aus dem Bild genommen, laufende bleiben ──
        import toolboxv2.mods.isaa.extras.icli2.app as appmod
        keep=appmod.MAX_BLOCKS; appmod.MAX_BLOCKS=3
        await pilot.press(*"long rest"); await pilot.press("enter"); await pilot.pause(0.4)
        live=[v for v in app.active.values()][0]
        await app._trim(); await pilot.pause(0.2)
        blocks=[w for w in app.chat.children if w.has_class("run")]
        assert len(blocks)<=3 and live.box in blocks, len(blocks)
        assert "ausgeblendet" in str(app.chat.children[0].render()) and all(c.is_attached for c in app.tool_cards)
        await app._trim(); assert "ausgeblendet" in str(app.chat.children[0].render())   # Hinweis bleibt beim naechsten Aufraeumen
        appmod.MAX_BLOCKS=keep
        await pilot.press("escape"); await pilot.press("escape"); await pilot.pause(0.6); assert not app.active
        print("render/steer/trim ok")
        # ── Schritt 8/9 ──
        await pilot.press(*"/login"); await pilot.press("enter"); await pilot.pause(0.3)
        assert app.user=="markin" and "markin" in str(app.query_one("#head").render())
        await pilot.press(*"/tb"); await pilot.press("enter"); await pilot.pause(0.3); assert isinstance(app.screen,P_)
        await pilot.press(*"vers"); await pilot.press("enter"); await pilot.pause(0.3); assert ("tb",{"target":"isaa.version"}) in fake.log
        await pilot.press(*"/tb"); await pilot.press("space"); await pilot.press(*"CloudM.get_user"); await pilot.press("space"); await pilot.press(*"a=1"); await pilot.press("enter"); await pilot.pause(0.3)
        assert ("tb",{"target":"CloudM.get_user","args":"a=1"}) in fake.log
        await pilot.press(*"/share"); await pilot.press("enter"); await pilot.pause(0.3); await pilot.press("enter"); await pilot.pause(0.3); assert ("share",{"share":"sh1"}) in fake.log
        await pilot.press(*"/voice"); await pilot.press("enter"); await pilot.pause(0.3); await pilot.press("enter"); await pilot.pause(0.3); assert ("voice",{"op":"omni"}) in fake.log
        await pilot.press("f4"); await pilot.pause(0.3); await pilot.press(*"vorher"); await pilot.press("f4"); await pilot.pause(0.3)
        assert app.prompt.text=="vorher gesprochener text", app.prompt.text
        app.prompt.set_text("")
        print("step8/9 ui ok")
        # ── Schritt 6/7 ──
        async def cmd(text, pick=None):
            await pilot.press(*[("space" if c==" " else c) for c in text]); await pilot.press("enter"); await pilot.pause(0.3)
            if pick is not None:
                assert isinstance(app.screen,P_), text
                if pick: await pilot.press(*pick); await pilot.pause(0.1)
                await pilot.press("enter"); await pilot.pause(0.3)
            assert not isinstance(app.screen,P_), text
        await cmd("/feature","")
        assert ("feature",{"name":"coder"}) in fake.log and app.features==["coder"] and "coder" in str(app.query_one("#head").render())
        await cmd("/feature execute"); assert ("feature",{"name":"execute"}) in fake.log
        await cmd("/skill",""); assert ("skill",{"name":"sk1"}) in fake.log
        await cmd("/mcp",""); assert ("mcp",{"name":"fs"}) in fake.log
        await cmd("/mcp git uvx mcp-server-git"); assert ("mcp_add",{"spec":"git uvx mcp-server-git"}) in fake.log
        await cmd("/plugin",""); assert ("plugin",{"name":"demo"}) in fake.log
        await cmd("/discord","conn"); assert ("discord",{"args":"connect"}) in fake.log
        await cmd("/discord send #a hallo"); assert ("discord",{"args":"send #a hallo"}) in fake.log
        await cmd("/telegram",""); assert ("telegram",{"args":"status"}) in fake.log
        await cmd("/plugin install owner/repo"); assert ("plugin_install",{"source":"owner/repo"}) in fake.log
        await cmd("/plugin confirm repo"); assert ("plugin_confirm",{"name":"repo"}) in fake.log
        await cmd("/plugin discard repo"); assert ("plugin_discard",{"name":"repo"}) in fake.log
        await cmd("/plugin remove repo"); assert ("plugin_remove",{"name":"repo"}) in fake.log
        await cmd("/plugin market owner/market"); assert ("market_add",{"source":"owner/market"}) in fake.log
        await cmd("/plugin market",""); assert ("plugin_install",{"source":"owner/tool"}) in fake.log
        await cmd("/plugin markets",""); assert ("market_remove",{"source":"owner/market"}) in fake.log
        n=len(fake.log); await cmd("/plugin install"); assert not [x for x in fake.log[n:] if x[0]!="remember"]
        await pilot.press(*"/runs"); await pilot.press("enter"); await pilot.pause(0.3)
        assert isinstance(app.screen,P_); await pilot.press("enter"); await pilot.pause(0.3)
        assert isinstance(app.screen,P_); await pilot.press("enter"); await pilot.pause(0.3)
        assert ("run",{"run":"self|r1","op":"reset"}) in fake.log, fake.log[-3:]
        print("step6/7 ui ok")
        # ── Vervollstaendigung beim Tippen ──
        await pilot.press("/", "s", "e"); await pilot.pause(0.1)
        assert app.completions[0][1]=="/session" and app.query_one("#complete").has_class("-open"), app.completions
        await pilot.press("tab"); await pilot.pause(0.3); assert app.prompt.text=="/session "
        assert [c[1] for c in app.completions]==["default","work"], app.completions   # Sessions aus der Engine
        app.prompt.set_text(""); await pilot.pause(0.1)
        # Unterbefehle: /agent s -> self, Tab uebernimmt, exakter Wert -> Enter fuehrt aus
        await pilot.press(*"/agent", "space", "s"); await pilot.pause(0.3)
        assert [c[1] for c in app.completions]==["self"], app.completions
        app.prompt.set_text("/agent c"); await pilot.pause(0.2); assert app.completions[0][1]=="coder", app.completions
        await pilot.press("tab"); await pilot.pause(0.1); assert app.prompt.text=="/agent coder" and not app.completions
        n=len(fake.log); await pilot.press("enter"); await pilot.pause(0.3); assert ("agent",{"name":"coder"}) in fake.log[n:]
        await pilot.press(*"/job", "space"); await pilot.pause(0.2)
        assert [c[1] for c in app.completions]==["fire","pause","remove"], app.completions
        await pilot.press("tab"); await pilot.pause(0.3); assert app.prompt.text=="/job fire "
        assert [(c[0],c[1]) for c in app.completions]==[("/job fire j1","nightly")], app.completions
        await pilot.press("tab"); await pilot.pause(0.1); n=len(fake.log)
        await pilot.press("enter"); await pilot.pause(0.3); assert ("job",{"job":"j1","op":"fire"}) in fake.log[n:], fake.log[n:]
        app.prompt.set_text("/model gate"); await pilot.pause(0.3); assert app.completions[0][0]=="/model gateway/big-model ", app.completions
        app.prompt.set_text(""); await pilot.pause(0.1)
        await pilot.press("/", "m", "o"); await pilot.pause(0.1); assert [c[1] for c in app.completions[:2]]==["/model","/mount"], app.completions
        await pilot.press("down"); await pilot.press("tab"); await pilot.pause(0.1); assert app.prompt.text=="/mount "
        app.prompt.set_text(""); await pilot.pause(0.1)
        await pilot.press("/", "r", "e", "s"); await pilot.pause(0.1); n=len(fake.log)
        await pilot.press("enter"); await pilot.pause(0.3); assert ("session_clear",{}) in fake.log[n:], fake.log[n:]   # Enter fuehrt Vorschlag aus
        await pilot.press("/", "x", "y", "z", "q"); await pilot.pause(0.1); await pilot.press("escape"); await pilot.pause(0.1)
        assert not app.query_one("#complete").has_class("-open"); app.prompt.set_text("")
        await pilot.press(*"hallo"); await pilot.pause(0.1); assert not app.completions; app.prompt.set_text(""); await pilot.pause(0.1)
        print("completion ok")
        # ── /vfs mit Pfad-Vervollstaendigung ──
        await pilot.press(*"/vfs"); await pilot.press("space"); await pilot.pause(0.3)
        assert [c[1] for c in app.completions]==["project/","notes.txt"], app.completions
        await pilot.press("tab"); await pilot.pause(0.3); assert app.prompt.text=="/vfs /project/"
        assert [c[1] for c in app.completions]==["src/","README.md"], app.completions      # naechste Ebene
        await pilot.press("s"); await pilot.pause(0.3); assert [c[1] for c in app.completions]==["src/"]
        await pilot.press("tab"); await pilot.pause(0.3); await pilot.press("m","a"); await pilot.pause(0.3)
        n=len(fake.log); await pilot.press("enter"); await pilot.pause(0.3)                   # Enter vervollstaendigt und zeigt
        assert ("vfs",{"path":"/project/src/main.py"}) in fake.log[n:], fake.log[n:]
        assert "INHALT /project/src/main.py" in [str(x.render()) for x in app.query(".note")][-1]
        await pilot.press(*"/vfs"); await pilot.press("space"); await pilot.press(*"/project/"); await pilot.pause(0.3)
        n=len(fake.log); await pilot.press("enter"); await pilot.pause(0.3); assert ("vfs",{"path":"/project/"}) in fake.log[n:]   # Ordner wie getippt
        # ── Picker: WASD und Pfeile, zurueck in die Palette ──
        await pilot.press("ctrl+p"); await pilot.pause(0.2); assert isinstance(app.screen,P_) and app.screen._title=="Befehle"
        await pilot.press("s","s","w"); await pilot.pause(0.1)
        from textual.widgets import OptionList
        assert app.screen.query_one(OptionList).highlighted==1                                 # /model
        await pilot.press("d"); await pilot.pause(0.4); assert app.screen._title.startswith("Modell")      # → oeffnet
        await pilot.press("left"); await pilot.pause(0.3); assert app.screen._title=="Befehle"             # ← zurueck
        await pilot.press("right"); await pilot.pause(0.4); assert app.screen._title=="Agent wechseln"
        await pilot.press("a"); await pilot.pause(0.3); assert app.screen._title=="Befehle"
        await pilot.press("slash","s","e","t","t"); await pilot.pause(0.2)                      # '/' startet Filter, s filtert
        assert app.screen.query_one(OptionList).option_count==1
        await pilot.press("enter"); await pilot.pause(0.4); assert app.screen._title=="Einstellungen"
        await pilot.press("enter"); await pilot.pause(0.3); assert app.prompt.text=="/set max_iterations " and len(app.screen_stack)==1
        await pilot.press(*"50"); await pilot.press("enter"); await pilot.pause(0.3); assert ("set",{"key":"max_iterations","value":"50"}) in fake.log
        await pilot.press("ctrl+p"); await pilot.pause(0.2); await pilot.press("escape"); await pilot.pause(0.2); assert len(app.screen_stack)==1
        # ── /context oeffnet die Seitenleiste und zeigt die Uebersicht ──
        assert not app.w_side.has_class("-open")
        await pilot.press(*"/context"); await pilot.press("enter"); await pilot.pause(0.6)
        side=str(app.w_side_text.render()); assert app.w_side.has_class("-open") and "System Prompt Total" in side and "1.8 %" in side, side[-300:]
        app.save_screenshot("/tmp/icli2_shot_context.svg")
        await pilot.press("f2"); await pilot.pause(0.2)
        # ── Mini-Fix: erste Taste nach verlorenem Fokus wird nicht verschluckt ──
        app.prompt.set_text(""); app.set_focus(None); await pilot.pause(0.1); assert not app.prompt.has_focus
        await pilot.press("h","i"); await pilot.pause(0.1); assert app.prompt.text=="hi", repr(app.prompt.text)
        app.prompt.set_text("")
        print("nav/vfs/context ok")
        await pilot.press(*"logs"); await pilot.press("enter"); await pilot.pause(0.8)
        app.save_screenshot("/tmp/icli2_shot.svg")
        await pilot.press(*"/q"); await pilot.press("enter")
    eng.send(P.SHUTDOWN); eng.join(5)
    assert not eng._thread.is_alive(); print("clean shutdown")
asyncio.run(main())
