param(
    [string]$RemoteRoot = '/root/autodl-tmp/final-online-grpo-20261012',
    [string]$LocalDirectory = 'D:/AgentLightning/mini-agent-rl/runs/final-online-grpo-20261012'
)
$ErrorActionPreference = 'Stop'
$taskKey = Join-Path $env:USERPROFILE '.ssh/autodl_ed25519'
$taskSshOptions = @('-i', $taskKey, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'StrictHostKeyChecking=accept-new', '-o', 'UserKnownHostsFile=NUL')
$taskHost = 'root@connect.cqa1.seetacloud.com'
New-Item -ItemType Directory -Force -Path $LocalDirectory | Out-Null
$taskReady = & ssh @taskSshOptions -p 41126 $taskHost "cat $RemoteRoot/results/archive-ready.json"
if ($LASTEXITCODE -ne 0) { throw 'Remote archive is not ready.' }
$taskReady | Set-Content -Encoding utf8 (Join-Path $LocalDirectory 'remote-archive-ready.json')
$taskMetadata = ($taskReady -join "`n") | ConvertFrom-Json
# Wrapper metadata is recorded first; the checksum sidecar is authoritative.
$taskArchive = "$RemoteRoot/results-results.tar.gz"
$taskArchiveName = 'results-results.tar.gz'
& scp @taskSshOptions -P 41126 "${taskHost}:$taskArchive" (Join-Path $LocalDirectory $taskArchiveName)
if ($LASTEXITCODE -ne 0) { throw 'Archive download failed; acknowledgement withheld.' }
& scp @taskSshOptions -P 41126 "${taskHost}:$taskArchive.sha256" (Join-Path $LocalDirectory "$taskArchiveName.sha256")
if ($LASTEXITCODE -ne 0) { throw 'Checksum download failed; acknowledgement withheld.' }
$taskExpected = ((Get-Content (Join-Path $LocalDirectory "$taskArchiveName.sha256") -Raw).Trim() -split '\s+')[0].ToLowerInvariant()
$taskActual = (Get-FileHash (Join-Path $LocalDirectory $taskArchiveName) -Algorithm SHA256).Hash.ToLowerInvariant()
if ($taskActual -ne $taskExpected) { throw 'Archive checksum mismatch; acknowledgement withheld.' }
& tar -tzf (Join-Path $LocalDirectory $taskArchiveName) | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Archive integrity check failed; acknowledgement withheld.' }
& tar -xzf (Join-Path $LocalDirectory $taskArchiveName) -C $LocalDirectory
if ($LASTEXITCODE -ne 0) { throw 'Archive extraction failed; acknowledgement withheld.' }
& ssh @taskSshOptions -p 41126 $taskHost "printf '%s\n' '$taskActual' > $RemoteRoot/results/local-received.ack"
if ($LASTEXITCODE -ne 0) { throw 'Archive verified locally, but remote acknowledgement failed.' }
Write-Output "Archive downloaded, SHA256 verified, extracted, and acknowledged: $taskActual"
