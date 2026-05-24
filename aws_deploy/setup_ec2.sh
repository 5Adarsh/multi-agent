#!/bin/bash
# AWS EC2 Amazon Linux 2023 Setup Script for ER-MAP Stack
# Run this script as root or with sudo: sudo bash aws_deploy/setup_ec2.sh

set -e

echo "=========================================="
echo " Starting EC2 ER-MAP Stack Setup (Amazon Linux 2023)"
echo "=========================================="

# 1. Update and install system dependencies
echo ">>> Installing System Dependencies..."
dnf update -y
dnf install -y python3 python3-pip git curl wget tar

# 2. Install Java 17 (Corretto is the default on AL2023)
echo ">>> Installing Java 17..."
dnf install -y java-17-amazon-corretto

# 3. Install Jenkins
echo ">>> Installing Jenkins..."
wget -O /etc/yum.repos.d/jenkins.repo https://pkg.jenkins.io/redhat-stable/jenkins.repo
rpm --import https://pkg.jenkins.io/redhat-stable/jenkins.io-2023.key

dnf install -y jenkins

# Ensure Jenkins runs on port 8080 (default)
systemctl enable jenkins
systemctl start jenkins

# 4. Setup Python Virtual Environment and Install Dependencies
echo ">>> Setting up Python Virtual Environment..."
REPO_DIR=$(pwd)
VENV_DIR="/home/ec2-user/ermap_venv"

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv $VENV_DIR
fi

echo ">>> Installing Python dependencies..."
# Upgrade pip
$VENV_DIR/bin/pip install --upgrade pip

# Install required packages for MLflow, ER-MAP Dashboard, and GRPO training
$VENV_DIR/bin/pip install transformers peft accelerate datasets flask mlflow scikit-learn pandas numpy sentencepiece

# 5. Setup Systemd Services for MLflow and Dashboard
echo ">>> Configuring Systemd Services..."

# Copy service files to systemd directory
cp aws_deploy/mlflow.service /etc/systemd/system/
cp aws_deploy/ermap-dashboard.service /etc/systemd/system/

# Fix user and paths in service files to point to actual repo directory
sed -i "s|/path/to/repo|$REPO_DIR|g" /etc/systemd/system/mlflow.service
sed -i "s|/path/to/repo|$REPO_DIR|g" /etc/systemd/system/ermap-dashboard.service
sed -i "s|User=ubuntu|User=ec2-user|g" /etc/systemd/system/mlflow.service
sed -i "s|User=ubuntu|User=ec2-user|g" /etc/systemd/system/ermap-dashboard.service
sed -i "s|/home/ubuntu|/home/ec2-user|g" /etc/systemd/system/mlflow.service
sed -i "s|/home/ubuntu|/home/ec2-user|g" /etc/systemd/system/ermap-dashboard.service

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
