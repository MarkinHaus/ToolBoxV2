# ToolBoxV2 Registry - Benutzerhandbuch

**Version**: 1.1
**Stand**: 2026-04-28

---

## Inhaltsverzeichnis

1. [Über das Registry](#über-das-registry)
2. [Schnellstart](#schnellstart)
3. [Module verwalten (CloudM)](#module-verwalten-cloudm)
4. [Registry CLI](#registry-cli)
5. [Sichtbarkeit von Mods](#sichtbarkeit-von-mods)
6. [Publishing Kurzanleitung](#publishing-kurzanleitung)
7. [Troubleshooting](#troubleshooting)
8. [Häufige Fragen (FAQ)](#häufige-fragen-faq)

---

## Über das Registry

Die ToolBoxV2 Registry ist die zentrale Verwaltungsplattform für alle ToolBoxV2 Module (Mods). Hier findest du:

- **Mods**: Erweiterungen für die ToolBoxV2
- **Libraries**: Nützliche Code-Bibliotheken
- **Artifacts**: Kompilierte Binaries und Apps (SimpleCore Desktop, TB CLI)

---

## Schnellstart

### Voraussetzungen

```bash
# ToolBoxV2 muss installiert sein
pip install toolboxv2
tb --version
```

### Einloggen (für Contributors)

```bash
# Login über CloudM.Auth
tb registry login
```

---

## Module verwalten (CloudM)

Die primäre Art Module zu installieren und verwalten ist über das CloudM-Modul.

### Interaktiver Manager (empfohlen)

```bash
tb -c CloudM mods manager
```

### Module installieren

```bash
# Über CloudM
tb -c CloudM mods install <module-name>

# Shortcut
tb --install <module-name>
```

### Module auflisten

```bash
tb -c CloudM mods list
```

### Module aktualisieren

```bash
tb --update <module-name>
```

### Module entfernen

```bash
tb --remove <module-name>
```

### Config generieren (für Publishing)

```bash
tb -c CloudM mods gen-config <module-name>
```

---

## Registry CLI

Die Registry-CLI bietet direkten Zugriff auf die Registry-API.

### Mods suchen

```bash
# Suche nach Mods
tb registry search discord

# Alle Mods auflisten
tb registry list

# Nach Typ filtern und sortieren
tb registry list --type mod --sort downloads --limit 20
```

### Mod-Details anzeigen

```bash
# Mod-Infos
tb registry info CloudM

# Mit Versionshistorie
tb registry info CloudM --versions
```

### Mod herunterladen

```bash
# Neueste Version herunterladen
tb registry download CloudM

# Spezifische Version
tb registry download CloudM --version 2.0.0

# In bestimmten Ordner
tb registry download CloudM --output ./mods/
```

### Versionen auflisten

```bash
tb registry versions CloudM
```

### Aktueller User

```bash
# Anzeigen wer eingeloggt ist
tb registry whoami
```

### Health Check

```bash
# Registry-Status prüfen
tb registry health
```

---

## Sichtbarkeit von Mods

### Sichtbarkeitsstufen

| Stufe | Beschreibung | Zugriff |
|-------|-------------|---------|
| **Public** | Öffentlich sichtbar | Jeder kann downloaden |
| **Unlisted** | Nicht gelistet | Nur mit direktem Link/Namen |
| **Private** | Privat | Nur der Owner |

### Public Mods

```bash
# Öffentliche Mods sind in der Suche sichtbar
tb registry search utility

# Jeder kann sie herunterladen
tb registry download MyUtilityMod
```

### Unlisted Mods

```bash
# Nicht in der Suche gelistet
# Aber herunterladen mit Name funktioniert:
tb registry download MyUnlistedMod
```

### Private Mods

```bash
# Private Mods benötigen Authentifizierung
tb registry login

# Download nur für den Owner
tb registry download MyPrivateMod
# Andere bekommen: 403 Forbidden
```

---

## Publishing Kurzanleitung

### 1. Einloggen

```bash
tb registry login
```

### 2. Publisher werden

Ein Publisher-Account wird über die API erstellt:

```bash
# Via HTTP API
curl -X POST https://registry.simplecore.app/api/v1/auth/register-publisher \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-publisher",
    "display_name": "My Publisher",
    "email": "contact@example.com"
  }'
```

Oder über den interaktiven Manager:

```bash
tb -c CloudM mods manager
# → REGISTRY → Register as Publisher
```

### 3. Package erstellen und hochladen

```bash
# metadata.json erstellen:
# {
#   "name": "my-mod",
#   "display_name": "My Awesome Mod",
#   "package_type": "mod",
#   "version": "1.0.0",
#   "description": "What my mod does",
#   "visibility": "unlisted"
# }

# Package erstellen
tb registry publish ./my-mod --create --metadata metadata.json

# Version hochladen
tb registry publish ./my-mod --upload --metadata metadata.json
```

### 4. Sichtbarkeit festlegen

```bash
# Auf Public (erfordert verifizierten Publisher)
tb registry publish my-mod --visibility public

# Auf Unlisted (für Beta/Testing)
tb registry publish my-mod --visibility unlisted

# Auf Private
tb registry publish my-mod --visibility private
```

### 5. Verification beantragen

Für Public-Mods muss der Publisher verifiziert sein:

```bash
# Via HTTP API
curl -X POST https://registry.simplecore.app/api/v1/publishers/verify \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"method": "github", "data": {"username": "dein-github"}}'
```

Ein Admin prüft den Antrag und verifiziert den Publisher.

### 6. Mod verwalten

```bash
# Mod löschen (Vorsicht!)
tb registry delete my-mod

# Version zurückziehen
tb registry yank my-mod 1.0.0 --reason "Critical bug"
```

---

## Troubleshooting

### Authentifizierungsprobleme

```bash
# Nicht eingeloggt?
# Error: Authentication required
# Lösung:
tb registry login

# Token abgelaufen?
# Error: Token expired / 401
# Lösung:
tb registry login
```

### Download-Probleme

```bash
# Mod nicht gefunden?
# Error: Package not found
# Lösung: Name prüfen (case-sensitive!)
tb registry search mod-name

# Version nicht gefunden?
# Error: Version not found
# Lösung: Verfügbare Versionen prüfen
tb registry versions mod-name
```

---

## Häufige Fragen (FAQ)

### Wie werde ich Publisher?

1. Bei der Registry einloggen: `tb registry login`
2. Publisher registrieren via API (`POST /api/v1/auth/register-publisher`)
3. Optional: Verification beantragen für Public-Mods

### Was ist der Unterschied zwischen Public und Unlisted?

| Public | Unlisted |
|--------|----------|
| In Suche sichtbar | Nicht in Suche |
| Jeder kann downloaden | Jeder mit Name kann downloaden |
| Verification benötigt | Keine Verification nötig |

### Kann ich meinen Mod später von Private auf Public ändern?

Ja:
```bash
tb registry publish my-mod --visibility public
```

(Erfordert verifizierten Publisher.)

### Was ist der Unterschied zwischen `tb registry` und `tb -c CloudM mods`?

- `tb -c CloudM mods` — installiert/verwaltet Module lokal auf deinem System
- `tb registry` — interagiert direkt mit der Registry-API (suchen, hochladen, herunterladen)

Für alltägliches Modmanagement nutze `tb -c CloudM mods manager`.
Für Publishing und Registry-Verwaltung nutze `tb registry`.

---

## Weiterführende Links

- [Contributors Guide](CONTRIBUTORS_GUIDE.md) - Ausführliche Anleitung für Contributors
- [Developers Guide](DEVELOPERS_GUIDE.md) - Für System-Entwickler
- [API Reference](API_REFERENCE.md) - HTTP-API Endpunkte
- [GitHub Repository](https://github.com/MarkinHaus/ToolBoxV2) - Source Code

---

**Letzte Aktualisierung**: 2026-04-28
**Kontakt**: support@toolboxv2.app
