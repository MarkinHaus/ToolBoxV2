# ToolBoxV2 Registry - Dokumentation

Willkommen zur offiziellen Dokumentation der ToolBoxV2 Registry.

---

## Schnellnavigation

### Für Nutzer
- **[User Guide](USER_GUIDE.md)** - Mods finden, installieren und verwenden

### Für Contributors (Mod-Entwickler)
- **[Contributors Guide](CONTRIBUTORS_GUIDE.md)** - Mods veröffentlichen und verwalten

### Für Entwickler (Registry-Entwicklung)
- **[Developers Guide](DEVELOPERS_GUIDE.md)** - Registry setup, Architektur, Contributing

### API-Referenz
- **[API Reference](API_REFERENCE.md)** - HTTP-API Endpunkte und Beispiele

---

## Kurzeinführung

Die ToolBoxV2 Registry ist die zentrale Plattform für:
- **Mods**: Erweiterungen für ToolBoxV2
- **Libraries**: Nützliche Python-Bibliotheken
- **Artifacts**: Kompilierte Binaries und Tools (SimpleCore Desktop, TB CLI)

### Was kann ich hier?

| Rolle | Kann... | Siehe... |
|-------|---------|----------|
| **Nutzer** | Mods downloaden, installieren, updaten | [User Guide](USER_GUIDE.md) |
| **Contributor** | Mods veröffentlichen, verwalten | [Contributors Guide](CONTRIBUTORS_GUIDE.md) |
| **Entwickler** | Registry mitentwickeln, API nutzen | [Developers Guide](DEVELOPERS_GUIDE.md) |

---

## Erste Schritte

### 1. ToolBoxV2 installieren

```bash
# Mit pip
pip install toolboxv2

# Version prüfen
tb --version
```

### 2. Mods installieren (über CloudM)

```bash
# Interaktiver Manager (empfohlen)
tb -c CloudM mods manager

# Direkt installieren
tb -c CloudM mods install CloudM

# Oder Shortcut
tb --install CloudM
```

### 3. Registry CLI verwenden

```bash
# Mods suchen
tb registry search discord

# Mods auflisten
tb registry list

# Mod-Details anzeigen
tb registry info CloudM

# Mod herunterladen
tb registry download CloudM
```

---

## Mod-Sichtbarkeit verstehen

### Public
- In der Suche sichtbar
- Jeder kann downloaden
- Erfordert verifizierten Publisher

### Unlisted
- Nicht in der Suche
- Aber mit Link/Namen downloadbar
- Keine Verifizierung nötig

### Private
- Nur für den Owner
- Nicht in der Suche
- Nicht downloadbar (außer Owner)

---

## Admin-Zugang

Der erste Admin wird direkt auf dem Registry-Server mit dem Admin-CLI erstellt:

```bash
# Auf dem RYZEN-Server ausführen
python admin_cli.py --db ./data/registry.db
# → Option 7: "Toggle admin"
```

Danach kann der Admin über die API Publisher verifizieren:

```bash
tb registry admin publisher verify --target <publisher-id>
```

Siehe [Developers Guide](DEVELOPERS_GUIDE.md#admin-management) für Details.

---

## Häufige Fragen

### Wie veröffentliche ich einen Mod?

Siehe [Contributors Guide](CONTRIBUTORS_GUIDE.md):

1. Einloggen: `tb registry login`
2. Publisher registrieren (über API oder CloudM)
3. Package erstellen: `tb registry publish ./my-mod --create --metadata metadata.json`
4. Version hochladen: `tb registry publish ./my-mod --upload --metadata metadata.json`

### Wie werde ich verifizierter Publisher?

1. Publisher registrieren (via `POST /api/v1/auth/register-publisher`)
2. Verification beantragen (via `POST /api/v1/publishers/verify`)
3. Admin verifiziert den Publisher

### Wie ändere ich die Sichtbarkeit?

```bash
tb registry publish my-mod --visibility public
tb registry publish my-mod --visibility unlisted
tb registry publish my-mod --visibility private
```

---

## Support

- **Issues**: [GitHub Issues](https://github.com/MarkinHaus/ToolBoxV2/issues)
- **Discord**: [ToolBoxV2 Discord](https://discord.gg/tb2)
- **Email**: support@toolboxv2.app

---

## Dokumentations-Status

| Dokument | Status | Letztes Update |
|----------|--------|----------------|
| [User Guide](USER_GUIDE.md) | ✅ Aktuell | 2026-04-28 |
| [Contributors Guide](CONTRIBUTORS_GUIDE.md) | ✅ Aktuell | 2026-04-28 |
| [Developers Guide](DEVELOPERS_GUIDE.md) | ✅ Aktuell | 2026-04-28 |
| [API Reference](API_REFERENCE.md) | ✅ Aktuell | 2026-04-28 |

---

## Versions-Historie

### v1.1.0 (2026-04-28)
- Dokumentation mit Code synchronisiert
- Admin-CLI und Bootstrapping dokumentiert
- Artifact Upload/Download Endpoints hinzugefügt
- Diff-System dokumentiert
- Versions-Endpoint dokumentiert
- CLI-Commands an tatsächlichen Code angepasst

### v1.0.0 (2025-02-25)
- Initiale Veröffentlichung
- CloudM.Auth Integration
- Public/Unlisted/Private Mods
- CLI und HTTP API

---

**Letzte Aktualisierung**: 2026-04-28
**Registry Version**: 1.1.0
