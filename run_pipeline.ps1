param(
    [switch]$AllowMock,
    [switch]$RunCalibration
)

$ErrorActionPreference = "Stop"

function Run-Step {
    param(
        [string]$Name,
        [string[]]$Args
    )

    Write-Host ""
    Write-Host "== $Name =="
    & python @Args
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

function Get-ModelProvider {
    param([string]$Model)

    $lower = $Model.ToLowerInvariant()
    if ($lower.StartsWith("glm-")) { return "zhipu" }
    if ($lower.StartsWith("gpt-")) { return "openai_compatible" }
    return "anthropic"
}

function Test-ProviderKey {
    param([string]$Provider)

    if ($Provider -eq "zhipu") {
        return -not [string]::IsNullOrWhiteSpace($env:ZHIPU_API_KEY)
    }
    if ($Provider -eq "openai_compatible") {
        return (-not [string]::IsNullOrWhiteSpace($env:CLIRELAY_API_KEY)) -or
               (-not [string]::IsNullOrWhiteSpace($env:OPENAI_API_KEY))
    }
    return -not [string]::IsNullOrWhiteSpace($env:ANTHROPIC_API_KEY)
}

function Get-KeyHint {
    param([string]$Provider)

    if ($Provider -eq "zhipu") { return "ZHIPU_API_KEY" }
    if ($Provider -eq "openai_compatible") { return "CLIRELAY_API_KEY or OPENAI_API_KEY" }
    return "ANTHROPIC_API_KEY"
}

$config = & python -c @"
import json, yaml
with open("config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
print(json.dumps({
    "models": cfg.get("models", []),
    "judge_model": cfg.get("judge_model"),
    "openai_base_url": cfg.get("openai_base_url"),
}))
"@
$config = $config | ConvertFrom-Json

if (-not $AllowMock) {
    foreach ($model in $config.models) {
        $provider = Get-ModelProvider $model
        if (-not (Test-ProviderKey $provider)) {
            throw "Missing key for generation model '$model'. Set $(Get-KeyHint $provider), or rerun with -AllowMock."
        }
    }

    $judgeProvider = Get-ModelProvider $config.judge_model
    if (-not (Test-ProviderKey $judgeProvider)) {
        throw "Missing key for judge model '$($config.judge_model)'. Set $(Get-KeyHint $judgeProvider), or rerun with -AllowMock."
    }
    if ($judgeProvider -eq "openai_compatible" -and [string]::IsNullOrWhiteSpace($config.openai_base_url)) {
        throw "Judge model '$($config.judge_model)' needs openai_base_url in config.yaml."
    }
}

Run-Step "Generate model outputs" @("src\run_eval.py")
Run-Step "Judge outputs" @("src\judge.py")
Run-Step "Build report" @("src\report.py")
Run-Step "Run unit tests" @("-m", "unittest", "discover", "-s", "tests")

if ($RunCalibration) {
    if ($AllowMock) {
        throw "Judge calibration requires a real judge model; do not combine -RunCalibration with -AllowMock."
    }
    Run-Step "Calibrate judge" @("src\calibrate_judge.py")
}

$generations = Get-Content -Path "results\generations.jsonl" -Encoding UTF8 |
    Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
    ForEach-Object { $_ | ConvertFrom-Json }
$scores = Get-Content -Path "results\scores.jsonl" -Encoding UTF8 |
    Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
    ForEach-Object { $_ | ConvertFrom-Json }
$pairwise = Get-Content -Path "results\pairwise.jsonl" -Encoding UTF8 |
    Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
    ForEach-Object { $_ | ConvertFrom-Json }

Write-Host ""
Write-Host "== Verification =="
Write-Host "generations.jsonl lines: $($generations.Count)"
Write-Host "scores.jsonl lines: $($scores.Count)"
Write-Host "pairwise.jsonl lines: $($pairwise.Count)"
Write-Host "report.md exists: $([System.IO.File]::Exists((Resolve-Path 'results\report.md')))"

if ($pairwise.Count -eq 0) {
    throw "pairwise.jsonl is empty. Pairwise evaluation did not run."
}

$generationModes = @($generations | ForEach-Object { $_.run_mode } | Sort-Object -Unique)
$judgeModes = @($scores | ForEach-Object { $_.judge_run_mode } | Sort-Object -Unique)
Write-Host "generation modes: $($generationModes -join ', ')"
Write-Host "judge modes: $($judgeModes -join ', ')"

if (-not $AllowMock) {
    if ($generationModes -contains "mock") {
        throw "Generation output contains mock records. Set the generation API key or rerun with -AllowMock."
    }
    if ($judgeModes -contains "mock") {
        throw "Judge output contains mock records. Set the judge API key or rerun with -AllowMock."
    }
}

Write-Host ""
Write-Host "Report: $(Resolve-Path 'results\report.md')"
