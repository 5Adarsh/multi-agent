#!/bin/bash
# AWS EC2 Amazon Linux 2023 Setup Script for ER-MAP Stack
# Supports: Amazon Linux 2023 (al2023)
# Run as root or with sudo: sudo bash aws_deploy/setup_ec2.sh

set -e

DNF=/usr/bin/dnf
SYSTEMCTL=/usr/bin/systemctl

echo "=========================================="
echo " ER-MAP Stack Setup — Amazon Linux 2023"
echo "=========================================="

# 1. Update package index
echo ">>> Updating package index..."
$DNF update -y

# 2. Install system dependencies
echo ">>> Installing system dependencies..."
$DNF install -y \
    python3 python3-pip python3-devel \
    git curl wget tar \
    ca-certificates

# 3. Install Java 17 (Amazon Corretto)
echo ">>> Installing Java 17 (Corretto)..."
$DNF install -y java-17-amazon-corretto-headless
java -version

# 4. Install Jenkins
echo ">>> Installing Jenkins..."
if ! $SYSTEMCTL is-active --quiet jenkins 2>/dev/null; then
    wget -O /etc/yum.repos.d/jenkins.repo \
        https://pkg.jenkins.io/redhat-stable/jenkins.repo
    rpm --import https://pkg.jenkins.io/redhat-stable/jenkins.io-2023.key
    $DNF install -y jenkins
fi
$SYSTEMCTL enable jenkins
$SYSTEMCTL start jenkins

# 5. Install Docker (using Amazon Linux 2023 built-in package)
echo ">>> Installing Docker..."
if ! command -v docker &>/dev/null; then
    $DNF install -y docker
fi
$SYSTEMCTL enable docker
$SYSTEMCTL start docker
# Allow ec2-user to run docker without sudo
usermod -aG docker ec2-user 2>/dev/null || true

# 6. Setup Python Virtual Environment
echo ">>> Setting up Python Virtual Environment..."
REPO_DIR=$(pwd)
VENV_DIR="/home/ec2-user/ermap_venv"

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi

echo ">>> Installing Python dependencies..."
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install \
    transformers peft accelerate datasets \
    flask mlflow scikit-learn pandas numpy \
    sentencepiece gymnasium groq fastapi \
    "uvicorn[standard]" pydantic python-dotenv

# 7. Setup Systemd Services for MLflow and Dashboard
echo ">>> Configuring Systemd Services..."

cp aws_deploy/mlflow.service /etc/systemd/system/
cp aws_deploy/ermap-dashboard.service /etc/systemd/system/

sed -i "s|/path/to/repo|$REPO_DIR|g" /etc/systemd/system/mlflow.service
sed -i "s|/path/to/repo|$REPO_DIR|g" /etc/systemd/system/ermap-dashboard.service
sed -i "s|User=ubuntu|User=ec2-user|g" /etc/systemd/system/mlflow.service
sed -i "s|User=ubuntu|User=ec2-user|g" /etc/systemd/system/ermap-dashboard.service
sed -i "s|/home/ubuntu|/home/ec2-user|g" /etc/systemd/system/mlflow.service
sed -i "s|/home/ubuntu|/home/ec2-user|g" /etc/systemd/system/ermap-dashboard.service

$SYSTEMCTL daemon-reload
$SYSTEMCTL enable mlflow
$SYSTEMCTL enable ermap-dashboard
$SYSTEMCTL restart mlflow
$SYSTEMCTL restart ermap-dashboard

echo "=========================================="
echo " Setup Complete!"
echo "=========================================="
JPWD=$(cat /var/lib/jenkins/secrets/initialAdminPassword 2>/dev/null || echo 'Not ready yet — wait 60s')
echo " - Jenkins:   Port 8080  (admin password: $JPWD)"
echo " - MLflow:    Port 5000"
echo " - Dashboard: Port 5050"
echo "=========================================="
echo "Next: see aws_deploy/jenkins_jobs_setup.md to configure Jenkins jobs."
