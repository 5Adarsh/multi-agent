# ============================================================
#  deploy_aws.ps1  -  Deploy ER-MAP Full Stack to AWS EC2
#  EC2: ec2-13-60-49-21.eu-north-1.compute.amazonaws.com
#  Usage: .\deploy_aws.ps1 -GroqApiKey "gsk_..."
# ============================================================

param(
    [Parameter(Mandatory=$true)]
    [string]$GroqApiKey,
    [string]$EC2Host   = "ec2-13-60-49-21.eu-north-1.compute.amazonaws.com",
    [string]$EC2User   = "ec2-user",
    [string]$PemFile   = ".\ml.pem",
    [string]$GitRepo   = "https://github.com/5Adarsh/multi-agent.git",
    [string]$RepoDir   = "/home/ec2-user/multi-agent"
)

$ErrorActionPreference = "Stop"

function Write-Step([string]$msg) {
    Write-Host "`n==> $msg" -ForegroundColor Cyan
}

# ── 0. Validate PEM file ──────────────────────────────────
Write-Step "Checking PEM key file..."
if (-not (Test-Path $PemFile)) {
    Write-Host "ERROR: PEM file not found at $PemFile" -ForegroundColor Red
    exit 1
}
icacls $PemFile /inheritance:r /grant:r "${env:USERNAME}:(R)" 2>$null | Out-Null
Write-Host "PEM file OK: $PemFile" -ForegroundColor Green

# ── 1. Test SSH connectivity ──────────────────────────────
Write-Step "Testing SSH connectivity to $EC2Host..."
try {
    $sshTest = ssh -i $PemFile -o ConnectTimeout=10 -o StrictHostKeyChecking=no `
        "${EC2User}@${EC2Host}" "echo SSH_OK" 2>&1 | Out-String
} catch {
    $sshTest = $_.Exception.Message
}
if ($sshTest -notmatch "SSH_OK") {
    Write-Host "ERROR: Cannot connect to EC2. Check security group allows port 22." -ForegroundColor Red
    Write-Host "Output: $sshTest"
    exit 1
}
Write-Host "SSH connection: OK" -ForegroundColor Green

# ── 2. Write the remote bash script to a temp file ────────
Write-Step "Generating remote setup script..."
$TempScript = "$env:TEMP\ermap_ec2_setup.sh"

# Write using ASCII to avoid BOM / CRLF issues on Linux
$scriptContent = @"
#!/bin/bash
set -e
REPO_DIR="$RepoDir"
GIT_REPO="$GitRepo"
GROQ_API_KEY="$GroqApiKey"

echo "--- [1/5] Clone / pull repo ---"
if [ -d "`$REPO_DIR/.git" ]; then
    cd "`$REPO_DIR" && git pull origin main
else
    git clone "`$GIT_REPO" "`$REPO_DIR"
    cd "`$REPO_DIR"
fi
cd "`$REPO_DIR"

echo "--- [2/5] Write .env file ---"
cat > .env <<ENVEOF
GROQ_API_KEY=`$GROQ_API_KEY
GROQ_DOCTOR_API_KEY=`$GROQ_API_KEY
GROQ_NURSE_API_KEY=`$GROQ_API_KEY
GROQ_PATIENT_API_KEY=`$GROQ_API_KEY
GROQ_EMPATHY_JUDGE_API_KEY=`$GROQ_API_KEY
GROQ_MEDICAL_JUDGE_API_KEY=`$GROQ_API_KEY
ERMAP_DOCTOR_MODEL=llama-3.1-8b-instant
ERMAP_NURSE_MODEL=llama-3.1-8b-instant
ERMAP_PATIENT_MODEL=llama-3.1-8b-instant
ERMAP_EMPATHY_JUDGE_MODEL=llama-3.3-70b-versatile
ERMAP_MEDICAL_JUDGE_MODEL=llama-3.3-70b-versatile
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
PORT=7860
ENVEOF
chmod 600 .env
echo ".env written OK"

echo "--- [3/5] Run EC2 setup script (Java + Jenkins + Python + systemd) ---"
sudo bash aws_deploy/setup_ec2.sh

echo "--- [4/5] Service status ---"
sudo systemctl status mlflow --no-pager || true
sudo systemctl status ermap-dashboard --no-pager || true
sudo systemctl status jenkins --no-pager || true

echo "--- [5/5] Deployment complete! ---"
PUBIP=`$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "13.60.49.21")
echo ""
echo "======================================================"
echo "  ER-MAP Stack is LIVE on AWS"
echo "======================================================"
echo "  Jenkins   ->  http://`$PUBIP:8080"
echo "  MLflow    ->  http://`$PUBIP:5000"
echo "  Dashboard ->  http://`$PUBIP:5050"
echo "  FastAPI   ->  http://`$PUBIP:7860"
echo "======================================================"
JPWD=`$(sudo cat /var/lib/jenkins/secrets/initialAdminPassword 2>/dev/null || echo "not ready yet - wait 60s and re-check")
echo "  Jenkins admin password: `$JPWD"
echo "======================================================"
"@

# Write with Unix line endings
[System.IO.File]::WriteAllText($TempScript, $scriptContent.Replace("`r`n", "`n"), [System.Text.Encoding]::ASCII)
Write-Host "Temp script created: $TempScript" -ForegroundColor Green

# ── 3. Copy script to EC2 and execute ─────────────────────
Write-Step "Uploading setup script to EC2..."
scp -i $PemFile -o StrictHostKeyChecking=no $TempScript "${EC2User}@${EC2Host}:/home/ec2-user/ermap_ec2_setup.sh"
Write-Host "Script uploaded." -ForegroundColor Green

Write-Step "Running full stack setup on EC2 (first run ~5 minutes)..."
ssh -i $PemFile -o StrictHostKeyChecking=no -t "${EC2User}@${EC2Host}" "chmod +x /home/ec2-user/ermap_ec2_setup.sh && bash /home/ec2-user/ermap_ec2_setup.sh"

# ── 4. Summary ─────────────────────────────────────────────
Write-Host "`n[deploy_aws.ps1] Deployment script finished!" -ForegroundColor Green
Write-Host "`nOpen in browser:" -ForegroundColor Yellow
Write-Host "  Jenkins:   http://13.60.49.21:8080" -ForegroundColor White
Write-Host "  MLflow:    http://13.60.49.21:5000" -ForegroundColor White
Write-Host "  Dashboard: http://13.60.49.21:5050" -ForegroundColor White
Write-Host "  FastAPI:   http://13.60.49.21:7860" -ForegroundColor White
