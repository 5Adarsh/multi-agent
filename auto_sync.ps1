$watcher = New-Object System.IO.FileSystemWatcher
$watcher.Path = $PSScriptRoot
$watcher.IncludeSubdirectories = $true
$watcher.EnableRaisingEvents = $true
$watcher.NotifyFilter = [System.IO.NotifyFilters]::LastWrite -bor [System.IO.NotifyFilters]::FileName -bor [System.IO.NotifyFilters]::DirectoryName

# Filter out .git folder to prevent infinite loops
$watcher.Filter = "*.*"

Write-Host "======================================================" -ForegroundColor Cyan
Write-Host " Auto-Sync is RUNNING!" -ForegroundColor Green
Write-Host " Any file you save in this folder will automatically" -ForegroundColor Yellow
Write-Host " be committed and pushed to GitHub." -ForegroundColor Yellow
Write-Host " Press Ctrl+C to stop." -ForegroundColor Red
Write-Host "======================================================" -ForegroundColor Cyan

$action = {
    $path = $Event.SourceEventArgs.FullPath
    $name = $Event.SourceEventArgs.Name
    $changeType = $Event.SourceEventArgs.ChangeType
    
    # Ignore changes in the .git folder, auto_sync itself, or python cache
    if ($path -match "\\\.git\\" -or $path -match "__pycache__" -or $name -eq "auto_sync.ps1") {
        return
    }

    Write-Host "`n[$((Get-Date).ToString('HH:mm:ss'))] Detected change: $name ($changeType)" -ForegroundColor Magenta
    Write-Host "Syncing to GitHub..." -ForegroundColor DarkGray
    
    # Wait a tiny bit to ensure the file is completely saved and released
    Start-Sleep -Milliseconds 500
    
    # Run Git commands
    git add .
    git commit -m "Auto-sync: $name was $changeType" | Out-Null
    
    $pushResult = git push origin main 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Successfully pushed to GitHub! 🚀" -ForegroundColor Green
    } else {
        Write-Host "Push failed. Make sure you have internet and no merge conflicts." -ForegroundColor Red
        Write-Host $pushResult -ForegroundColor DarkRed
    }
}

Register-ObjectEvent $watcher "Changed" -Action $action | Out-Null
Register-ObjectEvent $watcher "Created" -Action $action | Out-Null
Register-ObjectEvent $watcher "Deleted" -Action $action | Out-Null
Register-ObjectEvent $watcher "Renamed" -Action $action | Out-Null

# Keep the script running
try {
    while ($true) { Start-Sleep -Seconds 1 }
} finally {
    Unregister-Event -SourceIdentifier $watcher.Name
    $watcher.Dispose()
    Write-Host "`nAuto-Sync stopped." -ForegroundColor Red
}
