#!/bin/bash
# AWS EC2 Ubuntu 22.04 Setup Script for ER-MAP Stack
# Run this script as root or with sudo: sudo bash setup_ec2.sh

set -e

echo "=========================================="
echo " Starting EC2 ER-MAP Stack Setup"
echo "=========================================="

# 1. Update and install system dependencies
echo ">>> Installing System Dependencies..."
apt-get update -y
apt-get install -y python3 python3-pip python3-venv openjdk-17-jre git curl wget

# 2. Install Jenkins
echo ">>> Installing Jenkins..."
# Add Jenkins key and repository
curl -fsSL https://pkg.jenkins.io/debian-stable/jenkins.io-2023.key | tee /usr/share/keyrings/jenkins-keyring.asc > /dev/null
echo deb [signed-by=/usr/share/keyrings/jenkins-keyring.asc] https://pkg.jenkins.io/debian-stable binary/ | tee /etc/apt/sources.list.d/jenkins.list > /dev/null

apt-get update -y
apt-get install -y jenkins

# Ensure Jenkins runs on port 8080 (default)
systemctl enable jenkins
systemctl start jenkins

# 3. Setup Python Virtual Environment and Install Dependencies
echo ">>> Setting up Python Virtual Environment..."
# Assuming we run this from inside the repo root directory, let's use the repo path
REPO_DIR=$(pwd)
VENV_DIR="/home/ubuntu/ermap_venv"

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv $VENV_DIR
fi

echo ">>> Installing Python dependencies..."
# Upgrade pip
$VENV_DIR/bin/pip install --upgrade pip

# Install required packages for MLflow, ER-MAP Dashboard, and GRPO training
$VENV_DIR/bin/pip install transformers peft accelerate datasets flask mlflow scikit-learn pandas numpy sentencepiece

# 4. Setup Systemd Services for MLflow and Dashboard
echo ">>> Configuring Systemd Services..."

# Copy service files to systemd directory
cp aws_deploy/mlflow.service /etc/systemd/system/
cp aws_deploy/ermap-dashboard.service /etc/systemd/system/

# Update paths in service files to point to actual repo directory
sed -i "s|/path/to/repo|$REPO_DIR|g" /etc/systemd/system/mlflow.service
sed -i "s|/path/to/repo|$REPO_DIR|g" /etc/systemd/system/ermap-dashboard.service

# Reload systemd and start services
systemctl daemon-reload
systemctl enable mlflow
systemctl enable ermap-dashboard
systemctl restart mlflow
systemctl restart ermap-dashboard

echo "=========================================="
echo " Setup Complete!"
echo "=========================================="
echo "Services Status:"
echo " - Jenkins: Port 8080 (Initial password: $(cat /var/lib/jenkins/secrets/initialAdminPassword 2>/dev/null || echo 'Not ready yet'))"
echo " - MLflow: Port 5000"
echo " - ER-MAP Dashboard: Port 5050"
echo "=========================================="
echo "Please refer to aws_deploy/jenkins_jobs_setup.md to configure your Jenkins jobs."
