$ErrorActionPreference = 'Stop'

$project = 'C:\dev\context-aware-ar-fog\unity\CafogQuest'
$unity = 'C:\Program Files\Unity\Hub\Editor\6000.3.16f1\Editor\Unity.exe'
$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$diagnosticRoot = Join-Path 'C:\dev\context-aware-ar-fog' ('cafog-bee-diagnostic-' + $timestamp)
if (Test-Path $diagnosticRoot) {
    $suffix = 1
    do {
        $diagnosticRoot = Join-Path 'C:\dev\context-aware-ar-fog' ('cafog-bee-diagnostic-' + $timestamp + '-' + $suffix)
        $suffix++
    } while (Test-Path $diagnosticRoot)
}
$buildLog = Join-Path $diagnosticRoot 'build.log'
$monitorLog = Join-Path $diagnosticRoot 'process-monitor.log'
$dumpDirectory = Join-Path $diagnosticRoot 'dumps'
$apk = Join-Path $project 'Builds\Android\CAFoG_Quest_Dev.apk'
$stallSeconds = 600
$pollSeconds = 15
$snapshotSequence = 0

New-Item -ItemType Directory -Path $diagnosticRoot,$dumpDirectory -Force | Out-Null

function Write-MonitorLog {
    param([string]$Message)
    $line = '{0} {1}' -f (Get-Date -Format o), $Message
    Add-Content -LiteralPath $monitorLog -Value $line
    Write-Output $line
}

function Get-ProcessRecords {
    Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        ForEach-Object {
            $path = $_.ExecutablePath
            if (-not $path) {
                $runtime = Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue
                if ($runtime) {
                    $path = $runtime.Path
                }
            }
            [pscustomobject]@{
                ProcessId = [int]$_.ProcessId
                ParentProcessId = [int]$_.ParentProcessId
                Name = [string]$_.Name
                Path = [string]$path
                CommandLine = [string]$_.CommandLine
                StartTime = $null
            }
        } |
        ForEach-Object {
            $runtime = Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue
            if ($runtime) {
                $_.StartTime = $runtime.StartTime
            }
            $_
        }
}

function Get-Ancestry {
    param(
        [int]$ProcessId,
        [object[]]$Records
    )
    $ancestry = @()
    $current = $Records | Where-Object { $_.ProcessId -eq $ProcessId } | Select-Object -First 1
    while ($current) {
        $ancestry += $current.ProcessId
        if ($current.ParentProcessId -eq 0 -or $ancestry -contains $current.ParentProcessId) {
            break
        }
        $current = $Records | Where-Object { $_.ProcessId -eq $current.ParentProcessId } | Select-Object -First 1
    }
    return $ancestry
}

function New-ProcessIdentity {
    param(
        [object]$Record,
        [object[]]$Records,
        [int]$RootProcessId
    )
    [pscustomobject]@{
        ProcessId = $Record.ProcessId
        ParentProcessId = $Record.ParentProcessId
        Name = $Record.Name
        Path = $Record.Path
        CommandLine = $Record.CommandLine
        StartTime = $Record.StartTime
        Ancestry = @(Get-Ancestry -ProcessId $Record.ProcessId -Records $Records)
        RootProcessId = $RootProcessId
    }
}

function Get-OwnedProcessIdentities {
    param(
        [int]$RootProcessId,
        [hashtable]$KnownIdentities
    )
    $records = @(Get-ProcessRecords)
    $changed = $true
    while ($changed) {
        $changed = $false
        foreach ($record in $records) {
            if ($KnownIdentities.ContainsKey($record.ProcessId)) {
                continue
            }
            $ancestry = @(Get-Ancestry -ProcessId $record.ProcessId -Records $records)
            if ($ancestry -contains $RootProcessId) {
                $KnownIdentities[$record.ProcessId] = New-ProcessIdentity `
                    -Record $record -Records $records -RootProcessId $RootProcessId
                $changed = $true
            }
        }
        if ($changed) {
            $records = @(Get-ProcessRecords)
        }
    }
    return @($KnownIdentities.Values)
}

function Write-ProcessSnapshot {
    param([hashtable]$KnownIdentities)
    $records = @(Get-ProcessRecords)
    foreach ($identity in $KnownIdentities.Values | Sort-Object ProcessId) {
        $current = $records | Where-Object { $_.ProcessId -eq $identity.ProcessId } | Select-Object -First 1
        if (-not $current) {
            Write-MonitorLog ("Process exited: PID={0} Name={1}" -f $identity.ProcessId,$identity.Name)
            continue
        }
        $runtime = Get-Process -Id $current.ProcessId -ErrorAction SilentlyContinue
        if ($runtime) {
            Write-MonitorLog ("PID={0} PPID={1} Name={2} Path={3} CPU={4} MemoryMB={5:N1} Start={6:o} Cmd={7}" -f `
                $current.ProcessId,$current.ParentProcessId,$current.Name,$current.Path,
                $runtime.CPU,($runtime.WorkingSet64 / 1MB),$runtime.StartTime,$current.CommandLine)
        }
    }
}

function Test-ProcessIdentity {
    param(
        [object]$Identity,
        [object[]]$Records,
        [int]$RootProcessId
    )
    $current = $Records | Where-Object { $_.ProcessId -eq $Identity.ProcessId } | Select-Object -First 1
    if (-not $current) {
        return [pscustomobject]@{ Valid = $false; Reason = 'PID no longer exists.' }
    }
    $checks = @(
        @($current.ProcessId -eq $Identity.ProcessId, 'PID changed.'),
        @($current.Path -eq $Identity.Path, 'Executable path changed or is unavailable.'),
        @($current.CommandLine -eq $Identity.CommandLine, 'Command line changed.'),
        @($current.StartTime -eq $Identity.StartTime, 'Start time changed.'),
        @($current.ParentProcessId -eq $Identity.ParentProcessId, 'Parent PID changed.'),
        @((Get-Ancestry -ProcessId $current.ProcessId -Records $Records) -contains $RootProcessId, 'Diagnostic root ancestry no longer matches.')
    )
    foreach ($check in $checks) {
        if (-not $check[0]) {
            return [pscustomobject]@{ Valid = $false; Reason = $check[1] }
        }
    }
    return [pscustomobject]@{ Valid = $true; Reason = 'Identity verified.' }
}

function Stop-OwnedProcess {
    param(
        [object]$Identity,
        [int]$RootProcessId
    )
    $records = @(Get-ProcessRecords)
    $verification = Test-ProcessIdentity -Identity $Identity -Records $records -RootProcessId $RootProcessId
    if (-not $verification.Valid) {
        Write-MonitorLog ("Not terminating PID={0}: {1}" -f $Identity.ProcessId,$verification.Reason)
        return
    }
    Write-MonitorLog ("Termination approved only after identity verification for PID={0}." -f $Identity.ProcessId)
    Stop-Process -Id $Identity.ProcessId -ErrorAction Stop
}

function Find-DumpTool {
    $candidates = @(
        (Get-Command procdump.exe -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue),
        (Get-Command dotnet-dump.exe -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue),
        'C:\Sysinternals\procdump.exe',
        'C:\Program Files\Sysinternals\procdump.exe',
        'C:\Program Files\SysinternalsSuite\procdump.exe'
    )
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path $candidate)) {
            return (Get-Item $candidate).FullName
        }
    }
    return $null
}

function Capture-Dump {
    param([int]$ProcessId)
    $tool = Find-DumpTool
    if (-not $tool) {
        Write-MonitorLog 'No ProcDump or dotnet-dump executable found; no process dump captured.'
        return
    }

    $safeName = 'stalled-process-' + $ProcessId + '.dmp'
    $dumpPath = Join-Path $dumpDirectory $safeName
    if ((Split-Path $tool -Leaf) -ieq 'procdump.exe') {
        $command = "`"$tool`" -accepteula -ma $ProcessId `"$dumpPath`""
        Write-MonitorLog ("Dump command: {0}" -f $command)
        try {
            & $tool -accepteula -ma $ProcessId $dumpPath 2>&1 |
                ForEach-Object { Add-Content -LiteralPath $monitorLog -Value $_ }
            Write-MonitorLog ("Dump exit code: {0}; output path: {1}" -f $LASTEXITCODE,$dumpPath)
        } catch {
            Write-MonitorLog ("Dump failed: {0}; output path: {1}" -f $_.Exception.Message,$dumpPath)
        }
    } else {
        $command = "`"$tool`" collect --process-id $ProcessId --type Full --output `"$dumpPath`""
        Write-MonitorLog ("Dump command: {0}" -f $command)
        try {
            & $tool collect --process-id $ProcessId --type Full --output $dumpPath 2>&1 |
                ForEach-Object { Add-Content -LiteralPath $monitorLog -Value $_ }
            Write-MonitorLog ("Dump exit code: {0}; output path: {1}" -f $LASTEXITCODE,$dumpPath)
        } catch {
            Write-MonitorLog ("Dump failed: {0}; output path: {1}" -f $_.Exception.Message,$dumpPath)
        }
    }
}

function Snapshot-BeeTraces {
    param([string]$Reason)
    $script:SnapshotSequence++
    $beeDirectory = Join-Path $project 'Library\Bee'
    if (-not (Test-Path $beeDirectory)) {
        Write-MonitorLog ("Trace snapshot skipped; Bee directory does not exist. Reason={0}" -f $Reason)
        return
    }
    $snapshotTag = '{0}-{1}-{2}' -f $timestamp,$script:SnapshotSequence,$Reason
    $patterns = @('backend*.traceevents','buildprogram*.traceevents','tundra.log.json','Player*.dag*','TundraBuildState.state*')
    foreach ($pattern in $patterns) {
        Get-ChildItem -LiteralPath $beeDirectory -Filter $pattern -File -ErrorAction SilentlyContinue |
            ForEach-Object {
                $source = $_
                $destination = Join-Path $diagnosticRoot ($source.Name + '-' + $snapshotTag)
                try {
                    Copy-Item -LiteralPath $source.FullName -Destination $destination -ErrorAction Stop
                    Write-MonitorLog ("Trace copied: source={0} sourceTimestamp={1:o} size={2} destination={3}" -f `
                        $source.FullName,$source.LastWriteTime,$source.Length,$destination)
                } catch {
                    Write-MonitorLog ("Trace copy failed: source={0} destination={1} error={2}" -f `
                        $source.FullName,$destination,$_.Exception.Message)
                }
            }
    }
}

function Get-MeaningfulState {
    param(
        [hashtable]$KnownIdentities,
        [string]$PreviousPhase
    )
    $progressSignals = @()
    $runtimeSignals = @()
    $inputPath = Join-Path $project 'Library\Bee\artifacts\Android\il2cppOutput\UnityClassRegistration.cpp'
    $outputPath = Join-Path $project 'Library\Bee\artifacts\Android\libunity\arm64-v8a\UnityClassRegistration.o'
    foreach ($path in @($inputPath,$outputPath)) {
        if (Test-Path $path) {
            $item = Get-Item $path
            $progressSignals += '{0} timestamp={1:o} size={2}' -f $path,$item.LastWriteTime,$item.Length
        } else {
            $progressSignals += '{0} missing' -f $path
        }
    }
    foreach ($identity in $KnownIdentities.Values) {
        $runtime = Get-Process -Id $identity.ProcessId -ErrorAction SilentlyContinue
        if ($runtime) {
            $runtimeSignals += 'PID={0} Name={1} CPU={2} MemoryMB={3:N1}' -f `
                $identity.ProcessId,$identity.Name,$runtime.CPU,($runtime.WorkingSet64 / 1MB)
        }
    }
    $beeDirectory = Join-Path $project 'Library\Bee'
    $tracePatterns = @('backend*.traceevents','buildprogram*.traceevents','tundra.log.json','Player*.dag*','TundraBuildState.state*')
    foreach ($pattern in $tracePatterns) {
        Get-ChildItem -LiteralPath $beeDirectory -Filter $pattern -File -ErrorAction SilentlyContinue |
            Sort-Object Name |
            ForEach-Object {
                $progressSignals += 'Trace {0} timestamp={1:o} size={2}' -f $_.Name,$_.LastWriteTime,$_.Length
            }
    }
    $phase = $PreviousPhase
    if (Test-Path $buildLog) {
        $recent = Get-Content $buildLog -Tail 80 | Where-Object { $_ -match 'Starting:|DisplayProgressbar:|Android PostProcess task|ExitCode:|Build Finished|Internal build system' }
        if ($recent) {
            $phase = [string]$recent[-1]
        }
    }
    [pscustomobject]@{
        Phase = $phase
        ProgressSignature = ($progressSignals -join '; ')
        RuntimeDiagnostics = ($runtimeSignals -join '; ')
    }
}

Write-MonitorLog ("Diagnostic root: {0}" -f $diagnosticRoot)
Write-MonitorLog ("Build log: {0}" -f $buildLog)
Write-MonitorLog ("Stall threshold seconds: {0}" -f $stallSeconds)
Write-MonitorLog ("Unity command: {0} -batchmode -quit -nographics -projectPath `"{1}`" -executeMethod CafogPhase4Validation.BuildAndroid -logFile `"{2}`"" -f $unity,$project,$buildLog)

$arguments = @('-batchmode','-quit','-nographics','-projectPath',$project,'-executeMethod','CafogPhase4Validation.BuildAndroid','-logFile',$buildLog)
$unityProcess = Start-Process -FilePath $unity -ArgumentList $arguments -WorkingDirectory $project -PassThru
$knownIdentities = @{}
$initialRecords = @(Get-ProcessRecords)
$unityRecord = $initialRecords | Where-Object { $_.ProcessId -eq $unityProcess.Id } | Select-Object -First 1
if (-not $unityRecord) {
    throw "Unable to capture launched Unity process identity for PID $($unityProcess.Id)."
}
$knownIdentities[$unityProcess.Id] = New-ProcessIdentity -Record $unityRecord -Records $initialRecords -RootProcessId $unityProcess.Id
Write-MonitorLog ("Unity identity captured: PID={0} Path={1} Start={2:o} Cmd={3}" -f `
    $unityRecord.ProcessId,$unityRecord.Path,$unityRecord.StartTime,$unityRecord.CommandLine)

$lastMeaningful = Get-Date
$previousState = Get-MeaningfulState -KnownIdentities $knownIdentities -PreviousPhase 'Build started'
$lastProgressSignature = $previousState.ProgressSignature
$lastPhase = $previousState.Phase

try {
    while (-not $unityProcess.HasExited) {
        Start-Sleep -Seconds $pollSeconds
        $unityProcess.Refresh()
        Get-OwnedProcessIdentities -RootProcessId $unityProcess.Id -KnownIdentities $knownIdentities | Out-Null
        Write-ProcessSnapshot -KnownIdentities $knownIdentities

        $state = Get-MeaningfulState -KnownIdentities $knownIdentities -PreviousPhase $lastPhase
        if ($state.RuntimeDiagnostics) {
            Write-MonitorLog ("Runtime evidence (not progress): {0}" -f $state.RuntimeDiagnostics)
        }
        if ($state.ProgressSignature -ne $lastProgressSignature -or $state.Phase -ne $lastPhase) {
            $lastMeaningful = Get-Date
            $lastProgressSignature = $state.ProgressSignature
            $lastPhase = $state.Phase
            Write-MonitorLog ("Meaningful build state: phase={0}; timestamp={1:o}; signals={2}" -f `
                $lastPhase,$lastMeaningful,$state.ProgressSignature)
        }

        $stalledFor = ((Get-Date) - $lastMeaningful).TotalSeconds
        if ($stalledFor -ge $stallSeconds) {
            Write-MonitorLog ("Suspected stall after {0:N0} seconds without meaningful build progress. Last phase={1}; last progress={2:o}" -f `
                $stalledFor,$lastPhase,$lastMeaningful)
            Snapshot-BeeTraces -Reason 'stall'
            $backend = $knownIdentities.Values |
                Where-Object { $_.Name -in @('bee_backend.exe','AndroidPlayerBuildProgram.exe') } |
                Select-Object -First 1
            if ($backend) {
                Capture-Dump -ProcessId $backend.ProcessId
            } else {
                Write-MonitorLog 'No owned Bee or AndroidPlayerBuildProgram process found for dump capture.'
            }
            Write-MonitorLog 'No processes were terminated. Manual approval is required to stop the diagnostic build.'
            break
        }
    }
} finally {
    $unityProcess.Refresh()
    if ($unityProcess.HasExited) {
        Write-MonitorLog ("Unity actual exit code: {0}" -f $unityProcess.ExitCode)
    } else {
        Write-MonitorLog ("Unity remains running with PID {0}; no termination was attempted." -f $unityProcess.Id)
    }
    Write-MonitorLog 'Final owned-process snapshot:'
    Write-ProcessSnapshot -KnownIdentities $knownIdentities
    if (Test-Path $buildLog) {
        $finalLog = Join-Path $diagnosticRoot 'build-final.log'
        Copy-Item -LiteralPath $buildLog -Destination $finalLog -ErrorAction Stop
        Write-MonitorLog ("Final build log copied: source={0} destination={1}" -f $buildLog,$finalLog)
    }
    Snapshot-BeeTraces -Reason 'final'
    if (Test-Path $apk) {
        $apkItem = Get-Item $apk
        Write-MonitorLog ("APK exists: path={0} size={1} timestamp={2:o}" -f $apk,$apkItem.Length,$apkItem.LastWriteTime)
    } else {
        Write-MonitorLog ("APK does not exist: {0}" -f $apk)
    }
}
