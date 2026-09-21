<#
Creates mysql.password from a prompt or from -PlainPassword, then runs init_db.py.

Run (interactive):  .\setup_database.ps1
Or (one line):        .\setup_database.ps1 -PlainPassword "yourpassword"
Or Windows cmd:       setup_database.bat
Or cmd with password:  setup_database.bat "yourpassword"

For an empty MySQL password, put MYSQL_PASSWORD= in .env and remove MYSQL_PASSWORD_FILE, then run: python init_db.py
#>
param(
    [Parameter(Mandatory = $false)]
    [string]$PlainPassword
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root

if (-not (Test-Path (Join-Path $Root "venv\Scripts\Activate.ps1"))) {
    Write-Error "Virtualenv not found under venv\. Run: python -m venv venv; .\venv\Scripts\Activate.ps1; pip install -r requirements.txt"
    exit 1
}

if (-not (Test-Path (Join-Path $Root ".env"))) {
    Copy-Item (Join-Path $Root ".env.example") (Join-Path $Root ".env")
    Write-Host "Created .env from .env.example - edit MYSQL_USER or MYSQL_DB if needed."
}

$pwPath = Join-Path $Root "mysql.password"

if (-not [string]::IsNullOrWhiteSpace($PlainPassword)) {
    $plain = $PlainPassword.Trim()
}
else {
    $plain = (Read-Host "Enter MySQL password for the user configured in MYSQL_USER (.env)").Trim()
}

[IO.File]::WriteAllText($pwPath, $plain, [Text.UTF8Encoding]::new($false))
Write-Host "Wrote $pwPath (gitignored)"

& (Join-Path $Root "venv\Scripts\Activate.ps1")
python init_db.py
if ($LASTEXITCODE -ne 0) {
    Write-Error "init_db.py failed."
    exit $LASTEXITCODE
}
Write-Host "Database ready. Start dashboard with: streamlit run soc_dashboard.py"
