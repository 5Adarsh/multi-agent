# Jenkins Jobs Setup Guide

Follow these steps to configure your Jenkins jobs on the EC2 server (`http://<EC2-IP>:8080`).

## Prerequisites
1. Ensure the setup script (`aws_deploy/setup_ec2.sh`) has run successfully.
2. Log into Jenkins and complete the initial setup (install suggested plugins, create admin user).

## Job 1: ER-MAP-GRPO
This job runs the lightweight demo GRPO training pipeline on CPU.

1. Click **New Item**.
2. Enter item name: `ER-MAP-GRPO`
3. Select **Freestyle project** and click **OK**.
4. Scroll down to **Build Steps** -> **Add build step** -> **Execute shell**.
5. Paste the following command:
   ```bash
   cd /path/to/repo  # Update this to the actual repository path
   source /home/ec2-user/ermap_venv/bin/activate
   export PYTHONIOENCODING=utf-8
   export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
   python3 -m ER_MAP.training.train_grpo --model distilgpt2 --episodes 1
   ```
6. Click **Save**.

## Job 2: IRIS-ML-DEMO
This job runs the standalone, high-accuracy Random Forest classifier on the Iris dataset, tracking metrics via MLflow.

1. Click **New Item**.
2. Enter item name: `IRIS-ML-DEMO`
3. Select **Freestyle project** and click **OK**.
4. Scroll down to **Build Steps** -> **Add build step** -> **Execute shell**.
5. Paste the following command:
   ```bash
   cd /path/to/repo  # Update this to the actual repository path
   source /home/ec2-user/ermap_venv/bin/activate
   export PYTHONIOENCODING=utf-8
   export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
   python3 iris_mlflow_demo.py
   ```
6. Click **Save**.

## AWS Security Group Configuration (Reminder)
Ensure your EC2 instance's security group allows inbound traffic on the following TCP ports:
- **22** (SSH)
- **8080** (Jenkins)
- **5000** (MLflow)
- **5050** (ER-MAP Dashboard)
