param([string]$file)

$ErrorActionPreference = "Stop"

$token = $env:SUPABASE_MANAGEMENT_TOKEN
$ref = $env:SUPABASE_PROJECT_REF

if ([string]::IsNullOrWhiteSpace($token)) {
    throw "SUPABASE_MANAGEMENT_TOKEN is required. Set it in your environment before running migrations."
}

if ([string]::IsNullOrWhiteSpace($ref)) {
    throw "SUPABASE_PROJECT_REF is required. Set it in your environment before running migrations."
}

function Invoke-SupabaseSql {
    param([string]$sql)

    $body = @{ query = $sql } | ConvertTo-Json -Depth 5
    Invoke-RestMethod -Uri "https://api.supabase.com/v1/projects/$ref/database/query" `
        -Method POST `
        -Headers @{ "Authorization" = "Bearer $token"; "Content-Type" = "application/json" } `
        -Body $body
}

if ($file) {
    $sql = Get-Content $file -Raw
} else {
    # Run all migrations in order.
    $migrations = Get-ChildItem "$PSScriptRoot\..\supabase\migrations\*.sql" | Sort-Object Name
    foreach ($m in $migrations) {
        Write-Host "Running $($m.Name)..."
        $sql = Get-Content $m.FullName -Raw
        try {
            Invoke-SupabaseSql -sql $sql | Out-Null
            Write-Host "  OK" -ForegroundColor Green
        } catch {
            Write-Host "  ERROR: $($_.Exception.Message)" -ForegroundColor Red
            throw
        }
    }
    return
}

$res = Invoke-SupabaseSql -sql $sql
$res | ConvertTo-Json
