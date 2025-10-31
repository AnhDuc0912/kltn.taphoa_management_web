# Auto retrain + reload script for Windows
# Usage: run this script (PowerShell) on a schedule (Task Scheduler) or manually.
# It runs the offline trainer, waits for completion, then calls the Flask admin endpoint
# to ask the running app to reload the newest model.
# Requirements:
# - The Flask app must be running and listening on http://localhost:5000
# - Python must be on PATH and virtualenv activated if needed
# - The trainer writes versioned model + metadata to ./models (it does by default)

# Run trainer
Write-Host "Starting training..."
$trainCmd = "./tools/train_reinforce.py --out ./models/re_ranker.pt"
$proc = Start-Process -FilePath python -ArgumentList $trainCmd -Wait -NoNewWindow -PassThru
if ($proc.ExitCode -ne 0) {
    Write-Error "Trainer failed with exit code $($proc.ExitCode)"
    exit $proc.ExitCode
}
Write-Host "Trainer finished successfully. Looking for newest metadata..."

# Find newest metadata file produced by trainer
$meta = Get-ChildItem -Path ./models -Filter "re_ranker_*.json" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $meta) {
    Write-Warning "No metadata file found (expected re_ranker_*.json). You can still trigger reload manually."
    $metaPath = $null
} else {
    Write-Host "Found metadata: $($meta.Name)"
    $metaPath = $meta.FullName
}

# Trigger reload via admin endpoint, include model_path if we found metadata
$reloadUrl = "http://localhost:5000/admin/reload_re_ranker"
try {
    if ($metaPath) {
        $metaJson = Get-Content -Path $metaPath -Raw | ConvertFrom-Json
        $modelPath = $metaJson.model_path
        $body = @{ model_path = $modelPath } | ConvertTo-Json
        Write-Host "Requesting app to load model: $modelPath"
        $resp = Invoke-RestMethod -Uri $reloadUrl -Method Post -Body $body -ContentType 'application/json' -ErrorAction Stop
    } else {
        Write-Host "Requesting app to reload re-ranker (no explicit model_path) at $reloadUrl"
        $resp = Invoke-RestMethod -Uri $reloadUrl -Method Post -Body '{}' -ContentType 'application/json' -ErrorAction Stop
    }
    Write-Host "Reload response:`n"; $resp | ConvertTo-Json -Depth 3
} catch {
    Write-Error "Failed to call reload endpoint: $_"
    exit 1
}

Write-Host "Auto-retrain+reload completed."
