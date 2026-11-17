# Testen in ToolBoxV2 – Leitfaden und Anti-Muster

> Stand 2026-11-17. Verbindlich für jeden Contributor und jeden Agenten-Lauf.
> Anlass: Prüfung von `master` @ `c36c3dc`. Die neuen Konto- und Sync-Funktionen waren in den Unit-Tests grün (Konto 67 passed, Sync 229 passed), im Betrieb aber an mindestens sechs Stellen kaputt. Die Tests hatten genau die Stellen weggepatcht oder nicht ausgeführt, an denen die Fehler saßen.
> Arbeitsablauf für Agenten: `docs/devdocs/agent_arbeitsprotokoll.md`.

---

## 1. Das Anti-Muster: „Der Test patcht weg, was im Betrieb kaputt ist“

**Ein Test, der den fehlerhaften Teil durch ein Double ersetzt, kann diesen Fehler nie finden.** Er wird grün, beweist aber nur, dass der Rest um das Double herum funktioniert. Ein grüner Lauf solcher Tests ist kein Beleg für das Produkt.

### Belegte Fälle aus `c36c3dc`

| Nr. | Was der Test gemacht hat | Was im Betrieb kaputt war | Warum der Test es nicht finden konnte |
|---|---|---|---|
| 1 | `first_run._read_device_owner` per `mock.patch` ersetzt | `onboarding_state()` ruft die Funktion ohne `app` auf, `device_owner.json` wird relativ zum Arbeitsordner gesucht → nach dem Verknüpfen bleibt `needs_account` true | Die kaputte Funktion war gepatcht |
| 2 | `get_installation_state` gemockt, der Zustand vorgegeben | Die Kontopflicht hängt vom echten Profil und der echten Kontodatei ab | Der Zustand wurde behauptet, nicht erzeugt |
| 3 | `DataApiTransport` weggepatcht | Echte Antwortformen der Data-API, 401-Behandlung („offline“ statt „reauth_required“) | Die Strecke zwischen Gerät und Server lief nie |
| 4 | Alle Tests laufen aus dem Checkout | `CloudM/LiveSync` ist in `pyproject.toml` vom Paket ausgeschlossen. In der installierten App stirbt der Sync mit `ModuleNotFoundError` | Im Checkout liegt das Modul; das Paket wurde nie getestet |
| 5 | `test_needs_account_false_after_link` setzt kein Profil | – | Der Test ist ohne Profil immer grün, er kann nicht rot werden |
| 6 | `test_stubs.py` schreibt beim Import `sys.modules["toolboxv2.utils.system.types"] = <Stub>` | Andere Testdateien (`test_account_bridge`) bekommen den Stub und schlagen im Gesamtlauf fehl, einzeln sind sie grün | Globaler Zustand zwischen Testdateien |
| 7 | `server_worker.py:47-57`: zweimal derselbe Import im `try/except`, danach nur eine Warnung | Falsches Paket deklariert (`python-multipart` statt `multipart`), Datei-Uploads sind in **jeder** Installation abgeschaltet | Kein Test lädt eine Datei hoch; der Fehler wird zur Log-Zeile |

### Woran man das Anti-Muster erkennt

1. Das Patch-Ziel liegt in `toolboxv2.…`, also im eigenen Code, und zwar im Modul unter Test oder einem direkten Mitspieler.
2. Der Test behauptet einen Zustand (`return_value = {...}`), statt ihn herzustellen (Datei schreiben, Profil setzen).
3. Der Test bleibt grün, wenn man die Produktänderung, die er absichern soll, wieder zurücknimmt.
4. Der Test läuft nur aus dem Repo-Ordner, nie gegen das gebaute Paket.
5. Ein `except Exception: pass` oder eine reine Log-Warnung macht aus einem Fehler einen „Erfolg“.

---

## 2. Regeln

### Regel 1 – Echt vor Fake vor Stub vor Mock

Wenn möglich die **echte** Implementierung verwenden: echte Dateien im Temp-Ordner, echte `BlobStorage`, echte App-Instanz mit `TB_DATA_DIR` im Temp-Ordner, echter Worker auf einem freien Port. Geht das nicht (Netz, Kosten, Zeit), dann ein **Fake**: eine kleine, funktionierende In-Memory-Implementierung derselben Schnittstelle. Stubs (feste Rückgabe) und Mocks (Aufrufprüfung) nur, wenn weder echt noch Fake geht. So empfiehlt es *Software Engineering at Google*, Kapitel 13 („prefer realism over isolation“), und der Google Testing Blog warnt, dass Mocks die Logik des Codes doppeln und Refactoring erschweren.

### Regel 2 – Doubles nur an der Prozessgrenze

Ersetzt werden darf nur, was **außerhalb des eigenen Prozesses** liegt oder nicht deterministisch ist:

| erlaubt zu ersetzen | Stelle |
|---|---|
| Netz zu `simplecore.app` | `App.run_http` bzw. `app.session.fetch` – mit einem Fake, der die **echten Antwortformen** liefert (`Result.as_dict()`) |
| LLM-Anbieter | Router-Adapter (`llm_router`) |
| Uhr, Zufall | `time.time`, Zufallsquelle, mit festem Wert |
| Betriebssystem-Programme (crontab, winget, apt) | Aufruf des Programms, z. B. ein Schein-Programm im `PATH` |

**Nicht** ersetzt wird der Code unter Test und kein Modul aus `toolboxv2`, das im Betrieb zwischen Eingang und Grenze läuft. „Don't mock types you don't own“ (Google TotT) gilt dazu umgekehrt: Fremde Bibliotheken nicht direkt mocken, sondern hinter der eigenen Grenzfunktion.

### Regel 3 – Über den Eingang testen, den der Betrieb benutzt

Eine Route über HTTP gegen den echten Worker, eine CLI über den echten Befehl, eine Modulfunktion so, wie ihr Aufrufer sie aufruft (mit denselben Argumenten, z. B. **ohne** `app`, wenn der Aufrufer `app` nicht übergibt). Google TotT: *Prefer Testing Public APIs Over Implementation-Detail Classes*.

### Regel 4 – Zustand prüfen, nicht Aufrufe

Geprüft wird das Ergebnis, das der Nutzer sieht oder das auf der Platte liegt: Antwort, Datei, Datensatz. Nicht, welche interne Funktion wie oft aufgerufen wurde. Aufrufprüfungen nur an der Grenze („genau 1 `set_data` für 1 geändertes Feld“ ist richtig, weil es das Übertragungsverhalten prüft). Tests, die bei jedem Refactoring brechen, ohne dass sich Verhalten ändert, sind *Change-Detector-Tests* (Google TotT) und werden umgeschrieben.

### Regel 5 – Wenn doch ein Mock, dann streng

1. `autospec=True` (oder `create_autospec`): Der Mock scheitert wie das Original, wenn er falsch aufgerufen wird.
2. Dort patchen, wo nachgeschlagen wird (Python-Doku, „Where to patch“), und das Ziel als vollständigen Pfad schreiben.
3. Patch nur im Test (`with patch(...)`, Dekorator oder `addCleanup`), nie auf Modulebene.
4. Jeder Patch steht mit einem Satz Begründung im Test („Netzgrenze zu simplecore.app“).

### Regel 6 – Kein globaler Zustand

1. Kein `sys.modules[...] = ...` auf Modulebene. Wenn nötig: `with patch.dict(sys.modules, {...}):` innerhalb des Tests.
2. Umgebungsvariablen mit `patch.dict(os.environ, ...)` und `self.addCleanup(patcher.stop)` direkt nach `start()`.
3. Daten nur in `tempfile.mkdtemp()`; nie in den Checkout, nie ins Benutzerprofil, nie relativ zum Arbeitsordner.
4. Singletons (`get_app`, `Session`) vor und nach dem Test zurücksetzen (vorhanden: `tests/conftest.py`, `tests/a_util.py`).
5. Ein Test, der allein grün und im Gesamtlauf rot ist, ist ein Fehler, nicht „flaky“.

### Regel 7 – Jeder Test muss rot werden können (Gegenprobe)

Für jeden Bugfix-Test gilt: **vor** dem Fix ausführen → rot, mit der Fehlermeldung des Bugs; Fix anwenden → grün; Fix zurücknehmen (`git stash`) → wieder rot; Fix zurück → grün. Befehl und Ausgabe aller drei Läufe kommen in den Bericht. Ein Test, den man nicht rot bekommt, prüft nichts. (Das ist die Handform des Mutation Testing; für ganze Module kann `mutmut` das automatisch zeigen.)

### Regel 8 – Das Paket testen, nicht nur den Checkout

Wer `pyproject.toml`, Imports zwischen Modulen, `package-data` oder Einstiegspunkte ändert, prüft das **gebaute Paket**: Wheel bauen, in eine frische venv installieren, aus einem fremden Ordner mit `python -I` starten (der Checkout liegt dann nicht im `sys.path`) und den betroffenen Pfad ausführen. Muster aus der Python-Packaging-Diskussion: bauen, in frische Umgebung installieren, mit `python -I` testen.

```bash
python -m build --wheel -o /tmp/tbwheel
python -m venv /tmp/tbvenv && /tmp/tbvenv/bin/pip install "/tmp/tbwheel/toolboxv2-*.whl[cli,web,isaa,desktop]"
cd /tmp && /tmp/tbvenv/bin/python -I -c "import toolboxv2.mods.CloudM.LiveSync.data_api_sync"
```

### Regel 9 – Betriebstests für die Ziele

Für jedes Produktziel gibt es mindestens einen Test, der den echten Ablauf fährt: isolierte Umgebung (`HOME`, `TB_DATA_DIR`, `LOCALAPPDATA`/`APPDATA` umgebogen), `tb workers start` bzw. der Worker auf freiem Port, HTTP-Anfragen von außen, Prüfung des Ergebnisses, sauberes Stoppen. Kontopflicht z. B.: ohne Kontodatei liefert **jede** geschützte Route 403/Weiterleitung, nicht nur `/`. Diese Tests sind langsam; sie laufen in der CI in einem eigenen Schritt.

### Regel 10 – Fakes werden selbst geprüft

Ein Fake (z. B. In-Memory-Data-API) bekommt einen kleinen Vertragstest: Dieselben Fälle laufen gegen den Fake und gegen die echte Implementierung (lokal gegen die echte `CloudM.UserDataAPI` mit Temp-Datenbank). Weichen die Antworten ab, ist der Fake falsch. Google TotT: *Exercise Service Call Contracts in Tests*, *Fake Your Way to Better Tests*.

### Regel 11 – Fehler sind laut

Kein `except Exception: pass` im Produktcode. Wird ein Fehler bewusst behandelt, dann mit einem konkreten Typ und einem Zustand, den ein Test sehen kann (z. B. `state="error", error="..."`). Ein fehlendes Pflichtpaket bricht den Start ab oder schaltet die Funktion sichtbar ab, und dafür gibt es einen Test.

### Regel 12 – Zeitlimit

Jeder Test hat ein Zeitlimit (in der CI `timeout-minutes` am Job, für Tests mit Netz- oder Thread-Anteil ein Limit im Test). Ein hängender Test soll **einen** Test rot machen, nicht den Job nach 6 Stunden abbrechen (so geschehen: 3.10/ubuntu, Lauf 44002714075).

---

## 3. Prüfliste für jeden Test (Review)

| Nr. | Frage | muss |
|---|---|---|
| 1 | Welche Patches/Doubles gibt es? Jedes liegt an einer Prozessgrenze (Regel 2)? | ja |
| 2 | Wird ein Zustand behauptet statt hergestellt? | nein |
| 3 | Läuft der Test über denselben Eingang und mit denselben Argumenten wie der Betrieb? | ja |
| 4 | Gegenprobe gemacht: Fix zurück → rot (Regel 7)? Ausgabe im Bericht? | ja |
| 5 | Schreibt der Test nur in Temp-Ordner und räumt Umgebung/Singletons auf? | ja |
| 6 | Allein **und** im Gesamtlauf grün? | ja |
| 7 | Betrifft die Änderung Paket/Imports → Paket-Test (Regel 8) gelaufen? | ja |
| 8 | `unittest`, kein `skip`/`xfail`/`importorskip` als Ausweg? | ja |

---

## 4. Quellen

- Google Testing Blog, Testing on the Toilet: [Don't Overuse Mocks](https://testing.googleblog.com/2013/05/testing-on-toilet-dont-overuse-mocks.html), [Know Your Test Doubles](https://testing.googleblog.com/2013/07/testing-on-toilet-know-your-test-doubles.html), [Fake Your Way to Better Tests](https://testing.googleblog.com/2013/06/testing-on-toilet-fake-your-way-to.html), [Prefer Testing Public APIs Over Implementation-Detail Classes](https://testing.googleblog.com/2015/01/testing-on-toilet-prefer-testing-public.html), [Change-Detector Tests Considered Harmful](https://testing.googleblog.com/2015/01/testing-on-toilet-change-detector-tests.html), [Don't Mock Types You Don't Own](https://testing.googleblog.com/2020/07/testing-on-toilet-dont-mock-types-you.html), [Effective Testing](https://testing.googleblog.com/2014/05/testing-on-toilet-effective-testing.html), [Exercise Service Call Contracts in Tests](https://testing.googleblog.com/2018/11)
- *Software Engineering at Google*, Kapitel 13 „Test Doubles“ (Realismus vor Isolation, Fakes, Zustand statt Interaktion): [Zusammenfassung](https://study-notes-dge.pages.dev/engineering-practices/software-engineering-at-google/chapters/ch13-test-doubles)
- Martin Fowler: [UnitTest](https://martinfowler.com/bliki/UnitTest.html) (solitary/sociable, Begriffe von Jay Fields)
- Python-Doku: [unittest.mock](https://docs.python.org/3.15/library/unittest.mock.html) (Abschnitte „Where to patch“, „Autospeccing“)
- Python Discourse: [Integrationstests gegen frisch gebaute Pakete](https://discuss.python.org/t/a-test-design-pattern-to-reuse-integration-tests-on-newly-built-distribution-packages/17006) (bauen, frische Umgebung, `python -I`)
