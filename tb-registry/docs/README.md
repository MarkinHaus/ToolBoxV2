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
- **Artifacts**: Kompilierte Binaries und Tools

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
# Installer herunterladen
curl -sSL https://install.tb2.app | sh

# Oder mit pip
pip install toolboxv2
```

### 2. Registry konfigurieren

```bash
# Standard-Registry ist bereits konfiguriert
tb registry list

# Einloggen für Contributors
tb login
```

### 3. Mods installieren

```bash
# Mod installieren
tb install CloudM

# Oder aus Registry durchsuchen
tb registry search discord
tb install discord-mod
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

## Häufige Fragen

### Wie veröffentliche ich einen Mod?

Siehe [Contributors Guide](CONTRIBUTORS_GUIDE.md):

1. Publisher erstellen
2. Mod mit `my_mod.yaml` vorbereiten
3. `tb registry upload ./my-mod/`
4. Sichtbarkeit wählen

### Wie werde ich verifizierter Publisher?

```bash
tb publisher verify \
  --github "dein-github" \
  --website "https://deine-site.com"
```

### Wie ändere ich die Sichtbarkeit?

```bash
# Auf Public (nach Verification)
tb registry publish my-mod --visibility public

# Auf Unlisted
tb registry publish my-mod --visibility unlisted

# Auf Private
tb registry publish my-mod --visibility private
```

---

## Support

- **Issues**: [GitHub Issues](https://github.com/toolboxv2/registry/issues)
- **Discord**: [ToolBoxV2 Discord](https://discord.gg/tb2)
- **Email**: support@toolboxv2.app

---

## Dokumentations-Status

| Dokument | Status | Letztes Update |
|----------|--------|----------------|
| [User Guide](USER_GUIDE.md) | ✅ Aktuell | 2026-02-25 |
| [Contributors Guide](CONTRIBUTORS_GUIDE.md) | ✅ Aktuell | 2026-02-25 |
| [Developers Guide](DEVELOPERS_GUIDE.md) | ✅ Aktuell | 2026-02-25 |
| [API Reference](API_REFERENCE.md) | ✅ Aktuell | 2026-02-25 |

---

## Versions-Historie

### v1.0.0 (2025-02-25)
- Initiale Veröffentlichung
- CloudM.Auth Integration
- Public/Unlisted/Private Mods
- CLI und HTTP API

---

**Letzte Aktualisierung**: 2026-02-25
**Registry Version**: 1.0.0
