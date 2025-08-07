Add-Type -AssemblyName System.Windows.Forms

Write-Host "==============================="
Write-Host "   Streamlit App Installer"
Write-Host "==============================="

# CONFIGURATION
$envName = "statquickie"       # Update if your env name is different
$yamlFile = "__environment__.yml"
$appFile = "app.py"
$shortcutName = "Start Basic Stat Quickie App.lnk"

# 1. Detect Anaconda/Miniconda location
$possibleCondaPaths = @(
    "$env:USERPROFILE\anaconda3",
    "$env:USERPROFILE\miniconda3",
    "C:\ProgramData\Anaconda3",
    "C:\ProgramData\Miniconda3"
)
$condaRoot = $null
foreach ($path in $possibleCondaPaths) {
    if (Test-Path "$path\Scripts\activate.bat") {
        $condaRoot = $path
        break
    }
}
if (-not $condaRoot) {
    [System.Windows.Forms.MessageBox]::Show("Anaconda/Miniconda not found! Please install Anaconda or Miniconda first.","ERROR",[System.Windows.Forms.MessageBoxButtons]::OK,[System.Windows.Forms.MessageBoxIcon]::Error) | Out-Null
    Write-Host "[ERROR] Anaconda/Miniconda not found!"
    exit 1
}
$activateBat = "$condaRoot\Scripts\activate.bat"
Write-Host "[OK] Found Anaconda/Miniconda at: $condaRoot"

# 2. Check YAML file exists
if (!(Test-Path $yamlFile)) {
    [System.Windows.Forms.MessageBox]::Show("Cannot find $yamlFile in the current folder!","ERROR",[System.Windows.Forms.MessageBoxButtons]::OK,[System.Windows.Forms.MessageBoxIcon]::Error) | Out-Null
    Write-Host "[ERROR] $yamlFile not found!"
    exit 1
}
Write-Host "[OK] Found $yamlFile"

# 3. Create or update Conda environment
Write-Host "[STEP] Creating/updating Conda environment: $envName"
$cmdEnvCreate = "`"$activateBat`" && conda env create -f $yamlFile -n $envName"
$process = Start-Process cmd.exe -ArgumentList "/c $cmdEnvCreate" -Wait -NoNewWindow -PassThru
if ($process.ExitCode -ne 0) {
    Write-Host "[INFO] Attempting to update existing environment..."
    $cmdEnvUpdate = "`"$activateBat`" && conda env update -f $yamlFile -n $envName"
    $process2 = Start-Process cmd.exe -ArgumentList "/c $cmdEnvUpdate" -Wait -NoNewWindow -PassThru
    if ($process2.ExitCode -ne 0) {
        [System.Windows.Forms.MessageBox]::Show("Failed to create or update Conda environment! See PowerShell for errors.","ERROR",[System.Windows.Forms.MessageBoxButtons]::OK,[System.Windows.Forms.MessageBoxIcon]::Error) | Out-Null
        Write-Host "[ERROR] Failed to create or update Conda environment!"
        exit 1
    }
    Write-Host "[OK] Conda environment updated."
} else {
    Write-Host "[OK] Conda environment created."
}

# 4. Write the Streamlit launcher (start-streamlit-app.ps1)
$launcherScript = @"
Add-Type -AssemblyName System.Windows.Forms
`$activateBat = '$activateBat'
Start-Process "cmd.exe" -ArgumentList "/K `$activateBat $envName && streamlit run $appFile"
[System.Windows.Forms.MessageBox]::Show('Streamlit app is starting in a new window. To stop it, close that window or press Ctrl+C in it.','App Starting',[System.Windows.Forms.MessageBoxButtons]::OK,[System.Windows.Forms.MessageBoxIcon]::Information) | Out-Null
"@
Set-Content -Path "start-streamlit-app.ps1" -Value $launcherScript -Encoding UTF8
Write-Host "[OK] Created 'start-streamlit-app.ps1' for double-click launching."

# 5. Optional: Create a Windows shortcut on Desktop
$desktop = [Environment]::GetFolderPath("Desktop")
$shortcut = Join-Path $desktop $shortcutName
$wshell = New-Object -ComObject WScript.Shell
$sc = $wshell.CreateShortcut($shortcut)
$sc.TargetPath = "powershell.exe"
$sc.Arguments = "-ExecutionPolicy Bypass -File `"$PWD\start-streamlit-app.ps1`""
$sc.WorkingDirectory = "$PWD"
$sc.WindowStyle = 1
$iconPath = "$PWD\icon.ico"
if (Test-Path $iconPath) {
    $sc.IconLocation = $iconPath
} else {
    # fallback to conda icon if custom icon not found
    $sc.IconLocation = "$condaRoot\Menu\AnacondaNavigator.ico"
}
$sc.Save()
Write-Host "[OK] Desktop shortcut created: $shortcut"

# 6. Generate Uninstaller Script
# Write the uninstall script as a here-string (no $ in front of variable names inside the here-string)
$uninstallerScript = @"
Add-Type -AssemblyName System.Windows.Forms

# 1. Remove run app launcher
`$launcher = Join-Path `$PSScriptRoot 'start-streamlit-app.ps1'
if (Test-Path `$launcher) {
    Remove-Item `$launcher -Force
    Write-Host '[OK] Removed launcher: start-streamlit-app.ps1'
}

# 2. Remove desktop shortcut
`$desktop = [Environment]::GetFolderPath('Desktop')
`$shortcutQuoted = "$shortcut"
if (Test-Path `$shortcutQuoted) {
    Remove-Item `$shortcutQuoted -Force
    Write-Host '[OK] Removed desktop shortcut.'
}

# 3. Remove Conda environment
`$envName = "$envName"   # Change if your env is named differently

# Try to detect conda root
`$possibleCondaPaths = @(
    "`$env:USERPROFILE\anaconda3",
    "`$env:USERPROFILE\miniconda3",
    "C:\ProgramData\Anaconda3",
    "C:\ProgramData\Miniconda3"
)
`$condaRoot = `$null
foreach (`$path in `$possibleCondaPaths) {
    if (Test-Path "`$path\Scripts\activate.bat") {
        `$condaRoot = `$path
        break
    }
}

if (`$condaRoot) {
    `$activateBat = "`$condaRoot\Scripts\activate.bat"
    Write-Host "[STEP] Removing Conda environment: `$envName"
    `$cmdLine = "/c "`"`$activateBat"`" && conda env remove -n `$envName -y"
	Start-Process -FilePath cmd.exe -ArgumentList `$cmdLine -Wait
    Write-Host '[OK] Conda environment removed.'
} else {
    Write-Host '[WARN] Could not find Conda installation. Please remove the environment manually if needed.'
}

[System.Windows.Forms.MessageBox]::Show('Uninstallation complete!','Uninstall Success',[System.Windows.Forms.MessageBoxButtons]::OK,[System.Windows.Forms.MessageBoxIcon]::Information) | Out-Null
Write-Host 'Uninstallation complete!'
"@

# Write the script to a file
Set-Content -Path "uninstall-streamlit-app.ps1" -Value $uninstallerScript -Encoding UTF8
Write-Host "[OK] Created uninstaller: uninstall-streamlit-app.ps1"

[System.Windows.Forms.MessageBox]::Show("Installation complete! Double-click the desktop shortcut or 'start-streamlit-app.ps1' to launch the app.","Success",[System.Windows.Forms.MessageBoxButtons]::OK,[System.Windows.Forms.MessageBoxIcon]::Information) | Out-Null
Write-Host "Installation complete! Double-click the desktop shortcut or 'start-streamlit-app.ps1' to launch your app."
