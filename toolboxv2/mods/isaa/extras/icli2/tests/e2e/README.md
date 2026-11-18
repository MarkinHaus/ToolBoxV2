# Echter Durchlauf (isaa-Mod + Ersatz-LLM)

Braucht ein installiertes toolboxv2 mit `isaa`-Extra und `textual` – **kein litellm**: das Modell `gateway/fake` geht über den nativen Gateway-Adapter des CompletionRouter (`TB_LLM_GATEWAY_URL`). Zustand liegt unter `/tmp/e2e/state` (nie im echten Profil). Kein API-Schlüssel nötig:
`fake_llm.py` ist ein lokaler OpenAI-kompatibler Server, der `think`, `vfs_shell` und `final_answer` aufruft.

    mkdir -p /tmp/e2e && cp fake_llm.py e2e.py rows.py /tmp/e2e/ && cd /tmp/e2e
    python fake_llm.py &          # Port 8765
    python rows.py > rows.txt 2>&1; grep -E "^(BOOT|ROWS|DO) " rows.txt     # jede Liste und Aktion des Backends
    python plug.py > plug.txt 2>&1; grep -E "^PLUG" plug.txt   # Plugin installieren/aktivieren/entfernen (braucht /tmp/e2e/plug/demo-plugin, siehe Kopf von plug.py)
    python e2e.py  > out.txt  2>&1; grep -E "^(READY|RUN|PICK|NOTE|DO|COMPLETE|SIDE|ENGINE)" out.txt

Die Skripte benutzen feste Pfade unter `/tmp/e2e` (Linux/macOS). Unter Windows die Pfade in den drei Dateien anpassen.
