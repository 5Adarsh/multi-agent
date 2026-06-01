#!/bin/bash
# AWS EC2 Ubuntu Setup Script for ER-MAP Stack
# Supports: Ubuntu 20.04 / 22.04 / 24.04
# Run as root or with sudo: sudo bash aws_deploy/setup_ec2.sh

set -e

echo "=========================================="
echo " ER-MAP Stack Setup — Ubuntu EC2"
echo "=========================================="

# 1. Update package index
echo ">>> Updating package index..."
apt-get update -y

# 2. Install system dependencies
echo ">>> Installing system dependencies..."
apt-get install -y \
    python3 python3-pip python3-venv python3-dev \
    git curl wget tar \
    ca-certificates gnupg lsb-release \
    software-properties-common

# 3. Install Java 17 (OpenJDK)
echo ">>> Installing Java 17..."
apt-get install -y openjdk-17-jdk
java -version

# 4. Install Jenkins
echo ">>> Installing Jenkins..."
if ! systemctl is-active --quiet jenkins 2>/dev/null; then
    curl -fsSL https://pkg.jenkins.io/debian-stable/jenkins.io-2023.key | tee \
        /usr/share/keyrings/jenkins-keyring.asc > /dev/null
    echo "deb [signed-by=/usr/share/keyrings/jenkins-keyring.asc] \
        https://pkg.jenkins.io/debian-stable binary/" | tee \
        /etc/apt/sources.list.d/jenkins.list > /dev/null
    apt-get update -y
    apt-get install -y jenkins
fi
systemctl enable jenkins
systemctl start jenkins

# 5. Install Docker
echo ">>> Installing Docker..."
if ! command -v docker &> /dev/null; then
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
        gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
        https://download.docker.com/linux/ubuntu \
        $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
        tee /etc/apt/sources.list.d/docker.list > /dev/null
    apt-get update -y
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin
fi
systemctl enable docker
systemctl start docker
# Allow ubuntu/ec2-user to use docker without sudo
usermod -aG docker ubuntu 2>/dev/null || usermod -aG docker ec2-user 2>/dev/null || true

# 6. Setup Python Virtual Environment and Install Dependencies
echo ">>> Setting up Python Virtual Environment..."
REPO_DIR=$(pwd)
VENV_DIR="/home/ubuntu/ermap_venv"
ECU="ubuntu"
# Support both ubuntu and ec2-user (depending on AMI)
if id ec2-user &>/dev/null; then
    VENV_DIR="/home/ec2-user/ermap_venv"
    ECU="ec2-user"
fi

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv $VENV_DIR
fi

echo ">>> Installing Python dependencies..."
$VENV_DIR/bin/pip install --upgrade pip
$VENV_DIR/bin/pip install \
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
sed -i "s|User=ubuntu|User=$ECU|g" /etc/systemd/system/mlflow.service
sed -i "s|User=ubuntu|User=$ECU|g" /etc/systemd/system/ermap-dashboard.service
sed -i "s|/home/ubuntu|/home/$ECU|g" /etc/systemd/system/mlflow.service
sed -i "s|/home/ubuntu|/home/$ECU|g" /etc/systemd/system/ermap-dashboard.service

systemctl daemon-reload
systemctl enable mlflow
systemctl enable ermap-dashboard
systemctl restart mlflow
systemctl restart ermap-dashboard

echo "=========================================="
echo " Setup Complete!"
echo "=========================================="
echo "Services Status:"
JPWD=$(cat /var/lib/jenkins/secrets/initialAdminPassword 2>/dev/null || echo 'Not ready yet — wait 60s')
echo " - Jenkins:   Port 8080  (admin password: $JPWD)"
echo " - MLflow:    Port 5000"
echo " - Dashboard: Port 5050"
echo "=========================================="
echo "Next: see aws_deploy/jenkins_jobs_setup.md to configure Jenkins jobs."
