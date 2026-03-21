# ToolBoxV2 — Onboarding Guide

## First Start

Run `tb` — you'll be asked to pick your profile once:

| # | Profile   | When to pick                                      |
|---|-----------|---------------------------------------------------|
| 1 | Consumer  | You use one app/mod and just want it to run       |
| 2 | Homelab   | You run multiple mods, features, flows locally    |
| 3 | Server    | You manage IT infrastructure / distributed system |
| 4 | Business  | You need a quick health check, nothing more       |
| 5 | Developer | You build mods, features, or extend the core      |

After picking, a config wizard runs. You can skip it and re-run anytime:
```
tb manifest init
```

Change your profile later:
```
tb manifest set app.profile developer
```

---

## Per-Profile: What does bare `tb` do?

| Profile   | `tb` opens...                          |
|-----------|----------------------------------------|
| consumer  | GUI directly (`tb gui`)                |
| homelab   | Interactive dashboard                  |
| server    | ASCII service/node overview, then exit |
| business  | 3-line health summary, then exit       |
| developer | Interactive dashboard                  |

---

## Consumer

**You only need two things:**
```bash
tb                  # opens your app
tb fl status        # check for feature updates
```

**Update your mod:**
```bash
tb mods             # shows installed mods
tb manifest set ... # if you need to change a setting
```

---

## Homelab

**Daily workflow:**
```bash
tb                  # dashboard — overview of everything
tb services         # start/stop services
tb manifest show    # view full config
tb fl list          # list installed + available features
tb fl unpack web    # install a new feature pack
```

**Build your own mod:**
```bash
tb manifest list            # see feature status
tb manifest pack myfeature  # pack into ZIP for sharing
```

---

## Server / IT Admin

**`tb` gibt dir sofort:** Nodes, Services, Load — dann Exit.

**Gezielter eingreifen:**
```bash
tb status                   # detaillierter Service-Status
tb workers                  # Worker-Manager
tb manifest set database.mode CB   # DB-Mode wechseln
tb manifest validate        # Config prüfen
tb manifest apply           # Nginx/Sub-Configs neu generieren
tb --sm                     # Service-Manager Startup
```

**Distributed Management:**
```bash
tb manifest set nginx.server_name simplecore.app
tb manifest set app.environment production
tb manifest apply
```

---

## Business

**`tb` zeigt:** `✅ Healthy (3/3 services)` — fertig.

Mehr Detail bei Bedarf:
```bash
tb status
```

---

## Developer

**Mod entwickeln:**
```bash
tb                              # Dashboard
tb mods                         # Mod-Manager
tb manifest list                # Feature-Übersicht
tb manifest enable myfeature    # Feature aktivieren
tb manifest pack myfeature      # ZIP packen
tb registry                     # In Registry hochladen
```

**Config während Dev:**
```bash
tb manifest set app.environment development
tb manifest set app.debug true
tb manifest get app.log_level
```

**Core erweitern:**
```bash
tb fl unpack isaa   # ISAA-Feature entpacken
tb manifest init    # Wizard neu durchlaufen
```

---

## Cheat Sheet
```
tb                          → Profile-Aktion (s.o.)
tb manifest init            → Config-Wizard starten
tb manifest set <key> <val> → Einzelwert setzen (+ .env sync)
tb manifest get <key>       → Wert lesen
tb manifest validate        → Fehler prüfen
tb manifest apply           → Nginx + Sub-Configs schreiben
tb fl status                → Feature-Loader Status
tb fl unpack <name>         → Feature installieren
tb status                   → Service-Übersicht
tb services                 → Service-Manager CLI
tb registry                 → Registry CLI
tb manifest set app.profile <profil>  → Profil wechseln
```
