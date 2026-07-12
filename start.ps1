$Host.UI.RawUI.WindowTitle = "DreamStage Launcher"

$Root     = Split-Path -Parent $MyInvocation.MyCommand.Path
$Backend  = Join-Path $Root "backend"
$Frontend = Join-Path $Root "frontend"
$Venv     = Join-Path $Backend ".venv\Scripts"
$Pip      = Join-Path $Venv "pip.exe"
$Python   = Join-Path $Venv "python.exe"

function Write-Step($msg) { Write-Host "  $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "  [OK] $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "  [WARN] $msg" -ForegroundColor Yellow }
function Write-Err($msg)  { Write-Host "  [ERR] $msg" -ForegroundColor Red }

Clear-Host
Write-Host ""
Write-Host "  =================================================" -ForegroundColor Magenta
Write-Host "       DreamStage - AI Music Studio" -ForegroundColor White
Write-Host "  =================================================" -ForegroundColor Magenta
Write-Host ""

# ── Python check ──────────────────────────────────────────────────────────────
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Err "Python not installed. Get it from https://www.python.org/downloads/"
    Write-Warn "(Tick 'Add Python to PATH' during install)"
    Read-Host "  Press Enter to exit"; exit 1
}
$pyVer = & python --version 2>&1
Write-Ok "Python: $pyVer"

# ── Node.js check ─────────────────────────────────────────────────────────────
if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    Write-Err "Node.js not installed. Get it from https://nodejs.org/"
    Read-Host "  Press Enter to exit"; exit 1
}
$nodeVer = & node --version 2>&1
Write-Ok "Node: $nodeVer"

# ── Create .venv if missing ───────────────────────────────────────────────────
if (-not (Test-Path "$Python")) {
    Write-Host ""
    Write-Step "Creating Python virtual environment..."
    Set-Location $Backend
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { Write-Err "venv creation failed."; Read-Host; exit 1 }
    Write-Ok "Virtual environment created"
}

# ── Install / sync backend packages ──────────────────────────────────────────
$needsInstall = $false
$checkPkgs = @("fastapi", "librosa", "pedalboard", "scikit_learn", "pandas")
foreach ($pkg in $checkPkgs) {
    $result = & $Python -c "import $($pkg.Replace('-','_'))" 2>&1
    if ($LASTEXITCODE -ne 0) { $needsInstall = $true; break }
}

if ($needsInstall) {
    Write-Host ""
    Write-Step "Installing backend packages (2-5 min first time)..."
    & $Pip install -r "$Backend\requirements.txt" --quiet
    if ($LASTEXITCODE -ne 0) { Write-Err "pip install failed."; Read-Host; exit 1 }
    Write-Ok "Backend packages installed"
} else {
    Write-Ok "Backend packages ready"
}

# ── pyworld (optional, improves pitch correction) ────────────────────────────
$hasPyworld = & $Python -c "import pyworld" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Warn "pyworld not installed (formant-preserving pitch correction unavailable)"
    Write-Warn "To install: cd backend && .venv\Scripts\pip install pyworld"
} else {
    Write-Ok "pyworld (WORLD vocoder) ready"
}

# ── Validate .env (not just existence — catch placeholders + missing keys) ───
$EnvFile = Join-Path $Backend ".env"
if (-not (Test-Path $EnvFile)) {
    Write-Host ""
    Write-Err ".env file not found at backend\.env"
    Write-Warn "Copy backend\.env.example to backend\.env and fill in your keys."
    Write-Warn "The backend cannot start without it."
    Read-Host "  Press Enter to exit"; exit 1
}
$envRaw = Get-Content $EnvFile -Raw
$envProblems = @()
foreach ($required in @("SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY", "R2_ACCOUNT_ID",
                        "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "GROQ_API_KEY")) {
    if ($envRaw -notmatch "(?m)^$required=.+$") { $envProblems += "$required is missing" }
    elseif ($envRaw -match "(?m)^$required=REPLACE_ME") { $envProblems += "$required is a placeholder" }
}
if ($envProblems.Count -gt 0) {
    Write-Host ""
    foreach ($p in $envProblems) { Write-Warn "backend\.env: $p" }
    Write-Warn "Some features will fail until these are filled in."
} else {
    Write-Ok "backend\.env keys present"
}

# ── Audio assets (self-repair: soundfont re-download if missing) ─────────────
$Soundfont = Join-Path $Backend "app\assets\soundfont.sf2"
if (-not (Test-Path $Soundfont)) {
    Write-Warn "SoundFont missing - downloading GeneralUser GS (~31 MB)..."
    try {
        Invoke-WebRequest -Uri "https://raw.githubusercontent.com/mrbumpy409/GeneralUser-GS/main/GeneralUser-GS.sf2" `
            -OutFile $Soundfont -UseBasicParsing
        Write-Ok "SoundFont downloaded"
    } catch {
        Write-Warn "SoundFont download failed - beats fall back to synth sounds"
    }
} else {
    Write-Ok "SoundFont ready"
}
foreach ($asset in @("app\assets\drums\kits\trap\kick.wav", "app\assets\bass\808.wav")) {
    if (-not (Test-Path (Join-Path $Backend $asset))) {
        Write-Warn "Asset missing: backend\$asset (run tools\build_kits.py / build_808.py)"
    }
}

# ── Supabase auto-resume (free projects pause after ~1 week idle) ─────────────
# A paused project = DNS dead = every login and API call fails with confusing
# errors. Detect INACTIVE and resume it automatically before starting servers.
$mgmtToken = $null; $projectRef = $null
if ($envRaw -match "(?m)^SUPABASE_MANAGEMENT_TOKEN=(.+)$") { $mgmtToken = $Matches[1].Trim() }
if ($envRaw -match "(?m)^SUPABASE_PROJECT_REF=(.+)$")      { $projectRef = $Matches[1].Trim() }

if ($mgmtToken -and $projectRef -and $mgmtToken -notmatch "your-supabase") {
    Write-Host ""
    Write-Step "Checking Supabase project status..."
    try {
        $h = @{ Authorization = "Bearer $mgmtToken" }
        $proj = Invoke-RestMethod -Uri "https://api.supabase.com/v1/projects/$projectRef" `
            -Headers $h -TimeoutSec 15
        if ($proj.status -eq "ACTIVE_HEALTHY") {
            Write-Ok "Supabase: ACTIVE_HEALTHY"
        } elseif ($proj.status -eq "INACTIVE") {
            Write-Warn "Supabase project is PAUSED - resuming it now (takes ~2-3 min)..."
            try {
                Invoke-RestMethod -Uri "https://api.supabase.com/v1/projects/$projectRef/restore" `
                    -Method POST -Headers $h -TimeoutSec 30 | Out-Null
            } catch {}
            $resumed = $false
            for ($i = 0; $i -lt 12; $i++) {
                Start-Sleep -Seconds 20
                try {
                    $proj = Invoke-RestMethod -Uri "https://api.supabase.com/v1/projects/$projectRef" `
                        -Headers $h -TimeoutSec 15
                    Write-Step "  Supabase: $($proj.status)"
                    if ($proj.status -eq "ACTIVE_HEALTHY") { $resumed = $true; break }
                } catch {}
            }
            if ($resumed) { Write-Ok "Supabase resumed and healthy" }
            else { Write-Warn "Supabase still waking up - logins may fail for a few more minutes" }
        } else {
            Write-Warn "Supabase status: $($proj.status) - continuing anyway"
        }
    } catch {
        Write-Warn "Could not check Supabase status (offline?) - continuing"
    }
} else {
    Write-Warn "No SUPABASE_MANAGEMENT_TOKEN/PROJECT_REF in .env - skipping pause check"
}

# ── AI model status ───────────────────────────────────────────────────────────
$ModelsDir    = Join-Path $Backend "ai\models"
$GenreModel   = Join-Path $ModelsDir "genre_classifier.joblib"
$ValenceModel = Join-Path $ModelsDir "valence_regressor.joblib"

Write-Host ""
Write-Host "  -- AI Model Status --" -ForegroundColor DarkCyan
if (Test-Path $GenreModel) { Write-Ok "Genre classifier:   READY" }
else { Write-Warn "Genre classifier:   not trained (rule fallback; train via option 2/3)" }
if (Test-Path $ValenceModel) { Write-Ok "Valence regressor:  READY" }
else { Write-Warn "Valence regressor:  not trained (arithmetic fallback)" }
Write-Host ""

# ── Menu: auto-starts [1] after 5 seconds so a double-click needs no input ────
Write-Host "  [1] Start DreamStage (auto-starts in 5s)" -ForegroundColor Green
Write-Host "  [2] Start DreamStage + train AI models in background" -ForegroundColor Cyan
Write-Host "  [3] Train AI models only (no server start)" -ForegroundColor Yellow
Write-Host "  [4] Run backend health check" -ForegroundColor Gray
Write-Host ""
Write-Host "  Press a number key, or wait for auto-start... " -ForegroundColor White -NoNewline

$choice = "1"
$deadline = (Get-Date).AddSeconds(5)
while ((Get-Date) -lt $deadline) {
    if ($Host.UI.RawUI.KeyAvailable) {
        $key = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
        if ($key.Character -match "[1-4]") { $choice = [string]$key.Character }
        break
    }
    Start-Sleep -Milliseconds 150
}
Write-Host $choice
Write-Host ""

# ── Health check ──────────────────────────────────────────────────────────────
if ($choice -eq "4") {
    Write-Step "Running backend health check..."
    Set-Location $Backend
    & $Python -c "
from app.services.ml_analyzer import analyze_full_ml
from app.services.mixer import create_final_mix
from app.services.vocal_processor import process_vocal
from app.services.beat_synthesizer import generate_beat
from app.services.beat_generator import analyze_vocal_mood
print('[OK] All services import cleanly')

wav, genre = generate_beat(bars=2, attempt=1)
print(f'[OK] Beat synthesizer: generated {len(wav)} bytes, genre={genre}')

import pathlib
models_dir = pathlib.Path('ai/models')
for name in ['genre_classifier', 'valence_regressor', 'energy_classifier']:
    path = models_dir / f'{name}.joblib'
    print(f'[  ] {name}: ' + ('READY' if path.exists() else 'NOT TRAINED'))
" 2>&1
    Write-Host ""
    Read-Host "  Press Enter to exit"
    exit 0
}

# ── Train AI models ───────────────────────────────────────────────────────────
if ($choice -eq "3" -or $choice -eq "2") {
    Write-Host ""
    Write-Host "  =============================================" -ForegroundColor Cyan
    Write-Host "   AI Training Pipeline" -ForegroundColor White
    Write-Host "  =============================================" -ForegroundColor Cyan
    Write-Host ""

    $trainScript = @"
Set-Location '$Backend'
Write-Host ''
Write-Host '  [1/3] Downloading GTZAN dataset (~1.2 GB)...' -ForegroundColor Cyan
& '$Python' '$Backend\ai\dataset_pipeline\download.py' --datasets gtzan --data-dir '$Backend\ai\data'
if (`$LASTEXITCODE -ne 0) { Write-Host 'Download failed' -ForegroundColor Red; Read-Host; exit 1 }

Write-Host ''
Write-Host '  [2/3] Extracting audio features (~10 min)...' -ForegroundColor Cyan
`$env:DATA_DIR    = '$Backend\ai\data'
`$env:OUTPUT_DIR  = '$Backend\ai\features'
& '$Python' '$Backend\ai\dataset_pipeline\extract_features.py' --dataset gtzan --workers 2
if (`$LASTEXITCODE -ne 0) { Write-Host 'Feature extraction failed' -ForegroundColor Red; Read-Host; exit 1 }

Write-Host ''
Write-Host '  [3/3] Training classifiers (~5 min)...' -ForegroundColor Cyan
& '$Python' '$Backend\ai\training\train_classifiers.py' --models-dir '$Backend\ai\models'
if (`$LASTEXITCODE -ne 0) { Write-Host 'Training failed' -ForegroundColor Red; Read-Host; exit 1 }

Write-Host ''
Write-Host '  AI training complete! Models saved to backend/ai/models/' -ForegroundColor Green
Write-Host '  Restart DreamStage to activate the trained classifiers.' -ForegroundColor Yellow
Read-Host 'Press Enter to close'
"@

    if ($choice -eq "2") {
        Write-Step "Launching AI training in background window..."
        Start-Process powershell -ArgumentList "-NoExit", "-Command", $trainScript `
            -WorkingDirectory $Backend
        Write-Ok "Training started in background window"
        Start-Sleep -Seconds 2
    } else {
        $confirmTrain = Read-Host "  Start training now? (y/N)"
        if ($confirmTrain -ne "y" -and $confirmTrain -ne "Y") {
            Write-Host "  Cancelled." -ForegroundColor Yellow
            Read-Host "Press Enter to exit"; exit 0
        }
        Invoke-Expression $trainScript
        exit 0
    }
}

# ── Frontend packages ─────────────────────────────────────────────────────────
if (-not (Test-Path "$Frontend\node_modules\next")) {
    Write-Host ""
    Write-Step "Installing frontend packages (1-3 min first time)..."
    Set-Location $Frontend
    npm install --silent
    if ($LASTEXITCODE -ne 0) { Write-Err "npm install failed."; Read-Host; exit 1 }
    Write-Ok "Frontend packages installed"
} else {
    Write-Ok "Frontend packages ready"
}

# ── Detect already-running servers (double-launch protection) ─────────────────
function Test-Http($url) {
    try {
        $r = Invoke-WebRequest -Uri $url -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop
        return $r.StatusCode -eq 200
    } catch { return $false }
}
function Test-PortBound($port) {
    return [bool](Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)
}
# A port can be BOUND while the server is still warming up (HTTP not yet
# answering) — spawning a second instance then dies with WinError 10013.
# Treat bound as running.
$backendRunning  = (Test-Http "http://localhost:8000/health") -or (Test-PortBound 8000)
$frontendRunning = (Test-Http "http://localhost:3000") -or (Test-PortBound 3000)

# ── Start backend ─────────────────────────────────────────────────────────────
Write-Host ""
if ($backendRunning) {
    Write-Ok "Backend already running at http://localhost:8000 - reusing it"
} else {
    Write-Step "Starting backend  ->  http://localhost:8000"
    $backendCmd = "
`$Host.UI.RawUI.WindowTitle = 'DreamStage Backend'
Set-Location '$Backend'
Write-Host '  DreamStage Backend' -ForegroundColor Magenta
Write-Host '  http://localhost:8000' -ForegroundColor Cyan
Write-Host ''
& '$Venv\uvicorn.exe' app.main:app --reload --port 8000
"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCmd `
        -WorkingDirectory $Backend
    Start-Sleep -Seconds 3
}

# ── Start frontend ────────────────────────────────────────────────────────────
if ($frontendRunning) {
    Write-Ok "Frontend already running at http://localhost:3000 - reusing it"
} else {
    Write-Step "Starting frontend ->  http://localhost:3000"
    $frontendCmd = "
`$Host.UI.RawUI.WindowTitle = 'DreamStage Frontend'
Set-Location '$Frontend'
Write-Host '  DreamStage Frontend' -ForegroundColor Magenta
Write-Host '  http://localhost:3000' -ForegroundColor Cyan
Write-Host ''
npm run dev
"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $frontendCmd `
        -WorkingDirectory $Frontend
}

# ── Neural producer (ACE-Step) — optional GPU engine ─────────────────────────
# When this server is up, beat generation uses the neural producer (a music
# model that composes around the artist's exact performance). When it is not,
# the app silently falls back to the built-in synthesizer — never required.
$AceDir = Join-Path $Root "acestep"
$AcePython = Join-Path $AceDir ".venv\Scripts\python.exe"
if (Test-Path $AcePython) {
    if ((Test-Http "http://localhost:8001/health") -or (Test-PortBound 8001)) {
        Write-Ok "Neural producer already running (ACE-Step on :8001)"
    } else {
        Write-Step "Starting neural producer (ACE-Step) ->  http://localhost:8001"
        $aceCmd = "
`$Host.UI.RawUI.WindowTitle = 'DreamStage Neural Producer (ACE-Step)'
Set-Location '$AceDir'
`$env:PYTHONIOENCODING = 'utf-8'
`$env:ACESTEP_CONFIG_PATH = 'acestep-v15-base'
Write-Host '  DreamStage Neural Producer - ACE-Step 1.5' -ForegroundColor Magenta
Write-Host '  Model loads on the FIRST beat request (~60-90s once), then ~1 min per song.' -ForegroundColor Gray
Write-Host ''
& '$AcePython' -m acestep.api_server --port 8001 --lm-model-path acestep-5Hz-lm-0.6B
"
        Start-Process powershell -ArgumentList "-NoExit", "-Command", $aceCmd `
            -WorkingDirectory $AceDir
    }
} else {
    Write-Warn "Neural producer not installed (acestep\.venv missing) - synthesizer beats only"
}

# ── Wait for readiness, then open the app ─────────────────────────────────────
Write-Host ""
Write-Step "Waiting for servers to be ready..."

$backendReady = $backendRunning
for ($i = 0; $i -lt 20 -and -not $backendReady; $i++) {
    Start-Sleep -Seconds 2
    $backendReady = Test-Http "http://localhost:8000/health"
}
$frontendReady = $frontendRunning
for ($i = 0; $i -lt 20 -and -not $frontendReady; $i++) {
    Start-Sleep -Seconds 2
    $frontendReady = Test-Http "http://localhost:3000"
}

Write-Host ""
Write-Host "  =================================================" -ForegroundColor Magenta
if ($backendReady -and $frontendReady) {
    Write-Host "   DreamStage is running!" -ForegroundColor Green
} else {
    Write-Host "   DreamStage is starting up..." -ForegroundColor Yellow
    if (-not $backendReady)  { Write-Host "   (backend still warming up)" -ForegroundColor Gray }
    if (-not $frontendReady) { Write-Host "   (frontend still warming up)" -ForegroundColor Gray }
}
Write-Host ""
Write-Host "   App  -->  http://localhost:3000" -ForegroundColor White
Write-Host "   API  -->  http://localhost:8000" -ForegroundColor White
Write-Host "   Docs -->  http://localhost:8000/docs" -ForegroundColor Gray
Write-Host ""
Write-Host "   To STOP: close the backend and frontend windows." -ForegroundColor Yellow
Write-Host "  =================================================" -ForegroundColor Magenta
Write-Host ""

Start-Process "http://localhost:3000"

Read-Host "  Press Enter to close this launcher"
