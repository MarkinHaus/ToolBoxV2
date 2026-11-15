# verify_install_windows.ps1
# R9 acceptance probe: reads an existing installation, changes and deletes nothing.
# Runs under Windows PowerShell 5.1 (powershell.exe). ASCII only.
# Output: one line per check: "PASS|FAIL  <ID>  <observation>",
#         final line "RESULT: n PASS, m FAIL". Exit 0 only if 0 FAIL.
# Never prints the secret value.
param(
    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA "toolboxv2"),
    [string]$SaveBaseline = "",
    [string]$CompareBaseline = ""
)

$ErrorActionPreference = "Continue"
$script:_pass = 0
$script:_fail = 0

function Emit {
    param([string]$Status, [string]$Id, [string]$Obs)
    if ($Status -eq "PASS") { $script:_pass++ } else { $script:_fail++ }
    Write-Output ("{0}  {1}  {2}" -f $Status, $Id, $Obs)
}

function Read-Manifest {
    param([string]$Path)
    $m = @{}
    if ($Path -ne "" -and (Test-Path -LiteralPath $Path)) {
        Get-Content -LiteralPath $Path -Encoding UTF8 | ForEach-Object {
            if ($_ -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*"?(.*?)"?\s*$') {
                $m[$Matches[1]] = $Matches[2]
            }
        }
    }
    return $m
}

function HttpGet {
    param([string]$Url)
    $r = @{ code = 0; body = "" }
    try {
        $req = [System.Net.HttpWebRequest]::Create($Url)
        $req.Timeout = 15000
        $req.ReadWriteTimeout = 15000
        $resp = $req.GetResponse()
        $r.code = [int]$resp.StatusCode
        $sr = New-Object System.IO.StreamReader($resp.GetResponseStream())
        $r.body = $sr.ReadToEnd()
        $sr.Close()
        $resp.Close()
    } catch {
        $r.code = 0
        $r.body = ""
    }
    return $r
}

function Sha256Hex {
    param([string]$Text)
    $sha = [System.Security.Cryptography.SHA256]::Create()
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($Text)
    $h = $sha.ComputeHash($bytes)
    return ([System.BitConverter]::ToString($h)).Replace("-", "").ToLower()
}

# --- tree under install dir, excluding uv-tools ---
$rootFull = ""
if (Test-Path -LiteralPath $InstallDir) {
    $rootFull = (Resolve-Path -LiteralPath $InstallDir).Path.TrimEnd('\')
}
$tree = @()
$dirs = @()
if ($rootFull -ne "") {
    $tree = @(Get-ChildItem -LiteralPath $rootFull -Recurse -Force -File -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -notmatch '[\\/]uv-tools[\\/]' })
    $dirs = @(Get-ChildItem -LiteralPath $rootFull -Recurse -Force -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -notmatch '[\\/]uv-tools[\\/]' })
}

$envPath = ""
$envFiles = @()
$dataDirs = @()
$appDir = ""
$prof = ""
$jwt = ""
$envText = ""
$secretLine = ""
$mf = @{}
$coreBuild = ""
$tbVersion = ""
$feats = ""

if ($rootFull -eq "") {
    foreach ($v in @("V1", "V2", "V3", "V4", "V5", "V6", "V7")) {
        Emit FAIL $v "install dir missing"
    }
} else {
    # ---------- V1 manifest ----------
    $mf = Read-Manifest (Join-Path $rootFull "install.manifest")
    if ($mf.ContainsKey("core_build")) { $coreBuild = $mf["core_build"] }
    if ($mf.ContainsKey("tb_version")) { $tbVersion = $mf["tb_version"] }
    if ($mf.ContainsKey("features")) { $feats = $mf["features"] }
    if ($feats -eq "mini core cli web isaa desktop" -and $coreBuild -ne "") {
        Emit PASS "V1" ("manifest ok, core_build=" + $coreBuild)
    } else {
        Emit FAIL "V1" ("features='" + $feats + "' core_build='" + $coreBuild + "'")
    }

    # ---------- V2 exactly one .env in root ----------
    $envFiles = @($tree | Where-Object { $_.Name -eq ".env" })
    if ($envFiles.Count -eq 1 -and $envFiles[0].DirectoryName -ieq $rootFull) {
        Emit PASS "V2" "exactly one .env in install root"
    } elseif ($envFiles.Count -eq 0) {
        Emit FAIL "V2" "no .env found under install dir"
    } else {
        $list = ($envFiles | ForEach-Object { $_.FullName.Substring($rootFull.Length).TrimStart('\') }) -join ", "
        Emit FAIL "V2" ($envFiles.Count.ToString() + " .env files: " + $list)
    }
    $envPath = Join-Path $rootFull ".env"

    # ---------- V3 secret format ----------
    if (Test-Path -LiteralPath $envPath) {
        Get-Content -LiteralPath $envPath -Encoding UTF8 | ForEach-Object {
            if ($_ -match '^\s*TB_JWT_SECRET\s*=\s*(.*?)\s*$') { $jwt = $Matches[1] }
            if ($_ -match '^\s*TB_JWT_SECRET\s*=') { $secretLine = $_.TrimEnd() }
        }
        $envText = Get-Content -LiteralPath $envPath -Raw -Encoding UTF8
    }
    if ($jwt -match '^[0-9a-f]{128}$') {
        Emit PASS "V3" "TB_JWT_SECRET matches ^[0-9a-f]{128}$"
    } elseif ($jwt -eq "") {
        Emit FAIL "V3" "TB_JWT_SECRET line not found"
    } else {
        Emit FAIL "V3" ("TB_JWT_SECRET wrong format (len=" + $jwt.Length + ")")
    }

    # ---------- V4 no TB_DATA_DIR, no REQUIRED ----------
    if ($envText -eq "") {
        Emit FAIL "V4" ".env unreadable"
    } else {
        $why = @()
        if ($envText -match '(?m)^\s*TB_DATA_DIR\s*=') { $why += "has TB_DATA_DIR line" }
        if ($envText -match 'REQUIRED') { $why += "contains REQUIRED" }
        if ($why.Count -eq 0) { Emit PASS "V4" "no TB_DATA_DIR line, no REQUIRED" }
        else { Emit FAIL "V4" ($why -join "; ") }
    }

    # ---------- V5 exactly one .data with app layout ----------
    $dataDirs = @($dirs | Where-Object { $_.Name -eq ".data" })
    if ($dataDirs.Count -ne 1) {
        Emit FAIL "V5" ($dataDirs.Count.ToString() + " .data dirs found (need exactly 1)")
    } else {
        $kids = @(Get-ChildItem -LiteralPath $dataDirs[0].FullName -Force -Directory -ErrorAction SilentlyContinue)
        $names = @($kids | ForEach-Object { $_.Name })
        $hasBlob = $names -contains "blob_cache"
        $hasCli = $names -contains "cli_sessions"
        $appChild = @($kids | Where-Object { $_.Name -ne "blob_cache" -and $_.Name -ne "cli_sessions" })
        if ($hasBlob -and $hasCli -and $appChild.Count -ge 1) {
            $appDir = $appChild[0].FullName
            Emit PASS "V5" (".data has blob_cache, cli_sessions, app dir " + $appChild[0].Name)
        } else {
            Emit FAIL "V5" ("contents: " + ($names -join ","))
        }
    }

    # ---------- V6 site-packages clean ----------
    $spPkg = Join-Path $rootFull "uv-tools\toolboxv2\Lib\site-packages\toolboxv2"
    if (-not (Test-Path -LiteralPath $spPkg)) {
        Emit FAIL "V6" "site-packages\toolboxv2 not found under uv-tools"
    } else {
        $bad = @()
        foreach ($n in @(".data", ".config", ".info", ".env")) {
            if (Test-Path -LiteralPath (Join-Path $spPkg $n)) { $bad += $n }
        }
        if ($bad.Count -eq 0) { Emit PASS "V6" "no runtime state inside site-packages" }
        else { Emit FAIL "V6" ("found: " + ($bad -join ",")) }
    }

    # ---------- V7 tb-manifest with profile ----------
    $tmPath = Join-Path $rootFull "tb-manifest.yaml"
    if (-not (Test-Path -LiteralPath $tmPath)) {
        Emit FAIL "V7" "tb-manifest.yaml missing in install root"
    } else {
        $tmText = Get-Content -LiteralPath $tmPath -Raw -Encoding UTF8
        if ($tmText -match '(?m)^\s*profile\s*:\s*"?([A-Za-z0-9_\-]+)"?\s*$') {
            $prof = $Matches[1]
            Emit PASS "V7" ("profile=" + $prof)
        } else {
            Emit FAIL "V7" "no profile key in tb-manifest.yaml"
        }
    }
}

# ---------- V8 health ----------
$h = HttpGet "http://127.0.0.1:8467/health"
if ($h.code -eq 200 -and $h.body -match 'healthy') {
    Emit PASS "V8" "health 200 healthy"
} else {
    Emit FAIL "V8" ("core not running (status=" + $h.code + ")")
}

# ---------- V9 wizard state ----------
$w = HttpGet "http://127.0.0.1:8467/api/wizard/state"
if ($w.code -ne 200) {
    Emit FAIL "V9" ("wizard/state status=" + $w.code)
} else {
    $js = $null
    try { $js = $w.body | ConvertFrom-Json } catch { $js = $null }
    $fr = "?"
    $wp = "?"
    if ($js -ne $null) {
        if ($js.PSObject.Properties["first_run"]) { $fr = $js.first_run }
        if ($js.PSObject.Properties["profile"]) { $wp = $js.profile }
    }
    if ($fr -eq $false -and $wp -ne $null -and $wp -eq $prof -and $prof -ne "") {
        Emit PASS "V9" ("first_run=false profile=" + $wp)
    } else {
        Emit FAIL "V9" ("first_run=" + $fr + " profile=" + $wp + " (manifest profile=" + $prof + ")")
    }
}

# ---------- V10 UI routes ----------
$r1 = HttpGet "http://127.0.0.1:8467/"
$r2 = HttpGet "http://127.0.0.1:8467/setup"
$n1 = 0
if ($r1.body -ne "") { $n1 = [System.Text.Encoding]::UTF8.GetByteCount($r1.body) }
if ($r1.code -eq 200 -and $r2.code -eq 200 -and $n1 -gt 10000) {
    Emit PASS "V10" ("/ " + $n1 + " bytes, /setup 200")
} else {
    Emit FAIL "V10" ("/ status=" + $r1.code + " (" + $n1 + " bytes), /setup status=" + $r2.code)
}

# ---------- V11 local_ui.json ----------
$luiPath = Join-Path $env:APPDATA "toolboxv2\local_ui.json"
if (-not (Test-Path -LiteralPath $luiPath)) {
    Emit FAIL "V11" "local_ui.json missing in APPDATA\toolboxv2"
} else {
    $lui = $null
    try { $lui = Get-Content -LiteralPath $luiPath -Raw -Encoding UTF8 | ConvertFrom-Json } catch { $lui = $null }
    $idv = ""
    if ($lui -ne $null -and $lui.PSObject.Properties["install_dir"]) { $idv = [string]$lui.install_dir }
    if ($idv -ne "" -and $idv.TrimEnd('\') -ieq $rootFull) {
        Emit PASS "V11" "install_dir matches install dir"
    } else {
        Emit FAIL "V11" ("install_dir=" + $idv + " expected " + $rootFull)
    }
}

# ---------- V12 no litellm ----------
$py = ""
if ($rootFull -ne "") { $py = Join-Path $rootFull "uv-tools\toolboxv2\Scripts\python.exe" }
if ($py -eq "" -or -not (Test-Path -LiteralPath $py)) {
    Emit FAIL "V12" "venv python not found (uv-tools\toolboxv2\Scripts)"
} else {
    $null = & $py -c "import litellm" 2>&1
    if ($LASTEXITCODE -ne 0) { Emit PASS "V12" "import litellm fails (not installed)" }
    else { Emit FAIL "V12" "litellm is importable" }
}

# ---------- baseline snapshot data ----------
$fileList = @()
if ($appDir -ne "") {
    $fileList = @(Get-ChildItem -LiteralPath $appDir -Recurse -Force -File -ErrorAction SilentlyContinue |
        Where-Object { ($_.Name -notlike "*.db-wal") -and ($_.Name -notlike "*.db-shm") -and ($_.FullName -notmatch '[\\/]logs[\\/]') } |
        ForEach-Object {
            $rel = $_.FullName.Substring($appDir.Length).TrimStart('\')
            $o = New-Object PSObject
            $o | Add-Member -MemberType NoteProperty -Name "path" -Value $rel
            $o | Add-Member -MemberType NoteProperty -Name "size" -Value $_.Length
            $o
        })
}

if ($SaveBaseline -ne "") {
    $b = New-Object PSObject
    $b | Add-Member -MemberType NoteProperty -Name "secret_hash" -Value (Sha256Hex $secretLine)
    $b | Add-Member -MemberType NoteProperty -Name "profile" -Value $prof
    $b | Add-Member -MemberType NoteProperty -Name "core_build" -Value $coreBuild
    $b | Add-Member -MemberType NoteProperty -Name "tb_version" -Value $tbVersion
    $b | Add-Member -MemberType NoteProperty -Name "features" -Value $feats
    $b | Add-Member -MemberType NoteProperty -Name "files" -Value $fileList
    $b | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $SaveBaseline -Encoding UTF8
    Write-Output ("baseline saved: " + $SaveBaseline)
}

if ($CompareBaseline -ne "") {
    $b = $null
    try { $b = Get-Content -Raw -LiteralPath $CompareBaseline -Encoding UTF8 | ConvertFrom-Json } catch { $b = $null }
    if ($b -eq $null) {
        foreach ($u in @("U1", "U2", "U3", "U4", "U5")) { Emit FAIL $u "baseline unreadable" }
    } else {
        $curHash = Sha256Hex $secretLine
        if ($curHash -eq [string]$b.secret_hash) { Emit PASS "U1" "secret unchanged" }
        else { Emit FAIL "U1" "secret hash differs" }

        if ($prof -eq [string]$b.profile) { Emit PASS "U2" ("profile=" + $prof) }
        else { Emit FAIL "U2" ("profile now=" + $prof + " baseline=" + $b.profile) }

        if ($feats -eq [string]$b.features -and $coreBuild -ne [string]$b.core_build) {
            Emit PASS "U3" ("features equal, core_build " + $b.core_build + " -> " + $coreBuild)
        } else {
            Emit FAIL "U3" ("features_equal=" + ($feats -eq [string]$b.features) + " core_build " + $b.core_build + " -> " + $coreBuild)
        }

        if ($appDir -eq "") {
            Emit FAIL "U4" "no app dir under .data"
        } else {
            $missing = @()
            $shrunk = @()
            foreach ($f in @($b.files)) {
                $p = Join-Path $appDir ([string]$f.path)
                if (-not (Test-Path -LiteralPath $p)) { $missing += [string]$f.path; continue }
                $item = Get-Item -LiteralPath $p
                if ($item.Length -lt [long]$f.size) { $shrunk += ([string]$f.path + " " + $item.Length + "<" + $f.size) }
            }
            if ($missing.Count -eq 0 -and $shrunk.Count -eq 0) {
                Emit PASS "U4" (@($b.files).Count.ToString() + " baseline files present, none shrunk")
            } else {
                Emit FAIL "U4" ("missing: " + ($missing -join ",") + " shrunk: " + ($shrunk -join ","))
            }
        }

        if ($envFiles.Count -eq 1 -and $dataDirs.Count -eq 1) {
            Emit PASS "U5" "still exactly one .env and one .data"
        } else {
            Emit FAIL "U5" (".env=" + $envFiles.Count + " .data=" + $dataDirs.Count)
        }
    }
}

Write-Output ("RESULT: " + $script:_pass + " PASS, " + $script:_fail + " FAIL")
if ($script:_fail -gt 0) { exit 1 } else { exit 0 }
