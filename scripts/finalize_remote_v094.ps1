param(
    [string]$RemoteEndpoint = "root@connect.cqa1.seetacloud.com",
    [int]$Port = 44249,
    [string]$IdentityFile = "$env:USERPROFILE\.ssh\autodl_ed25519",
    [string]$RemoteRepo = "/root/autodl-tmp/mini-agent-rl",
    [string]$BackupRoot = "reports\v094-remote-backup",
    [string[]]$Artifacts = @("reports/v094-signal-diagnostic.json", "runs/v094-signal-diagnostic.db", "runs/v094-signal-diagnostic.jsonl"),
    [switch]$AllowLargeBackup,
    [switch]$Shutdown
)

$ErrorActionPreference = "Stop"
$quotedArtifacts = ($Artifacts | ForEach-Object { "'$_'" }) -join " "
$sizeText = & ssh -o BatchMode=yes -i $IdentityFile -p $Port $RemoteEndpoint "cd '$RemoteRepo' && du -cb $quotedArtifacts | tail -n 1 | cut -f1"
if ($LASTEXITCODE -ne 0 -or -not ($sizeText -match '^\d+$')) { throw "Unable to determine remote artifact size." }
$totalBytes = [int64]$sizeText
if ($totalBytes -gt 1000000000 -and -not $AllowLargeBackup) {
    throw "Backup is $totalBytes bytes (>1GB). Re-run only after explicit approval with -AllowLargeBackup."
}

$remoteArchive = "/tmp/mini-agent-rl-v094-backup.tar.gz"
$remoteHash = "/tmp/mini-agent-rl-v094-backup.sha256"
& ssh -o BatchMode=yes -i $IdentityFile -p $Port $RemoteEndpoint "cd '$RemoteRepo' && tar -czf '$remoteArchive' $quotedArtifacts && sha256sum '$remoteArchive' > '$remoteHash'"
if ($LASTEXITCODE -ne 0) { throw "Remote archive creation failed." }

New-Item -ItemType Directory -Force -Path $BackupRoot | Out-Null
& scp -i $IdentityFile -P $Port "${RemoteEndpoint}:$remoteArchive" "${RemoteEndpoint}:$remoteHash" $BackupRoot
if ($LASTEXITCODE -ne 0) { throw "Backup download failed." }

$archivePath = Join-Path $BackupRoot "mini-agent-rl-v094-backup.tar.gz"
$hashPath = Join-Path $BackupRoot "mini-agent-rl-v094-backup.sha256"
$expected = (Get-Content -LiteralPath $hashPath -Raw).Split(' ', [System.StringSplitOptions]::RemoveEmptyEntries)[0].ToLowerInvariant()
$actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $archivePath).Hash.ToLowerInvariant()
if ($actual -ne $expected) { throw "Downloaded archive SHA-256 mismatch." }

& ssh -o BatchMode=yes -i $IdentityFile -p $Port $RemoteEndpoint "rm -f '$remoteArchive' '$remoteHash'"
if ($Shutdown) {
    & ssh -o BatchMode=yes -i $IdentityFile -p $Port $RemoteEndpoint "/usr/bin/shutdown"
}
Write-Output "BACKUP_OK bytes=$totalBytes sha256=$actual shutdown=$($Shutdown.IsPresent)"

