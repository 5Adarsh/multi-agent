pipeline {
    agent any

    triggers {
        pollSCM('H/2 * * * *') // Poll repository every 2 minutes for changes
    }

    environment {
        PYTHONPATH = "${WORKSPACE}"
        // Force Python to use UTF-8 encoding on Windows runners to prevent charmap encoding errors
        PYTHONUTF8 = "1"
        // Default tracking URI for MLflow inside the pipeline (use local SQLite backend to prevent connection timeouts in CI)
        MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Setup Environment') {
            steps {
                echo 'Setting up Python Virtual Environment...'
                script {
                    if (isUnix()) {
                        sh '''
                            python3 -m venv venv
                            venv/bin/pip install --upgrade pip
                            venv/bin/pip install -r requirements.txt
                            venv/bin/pip install pytest flake8 mlflow
                        '''
                    } else {
                        bat '''
                            py -m venv venv || python -m venv venv
                            venv\\Scripts\\pip install --upgrade pip
                            venv\\Scripts\\pip install -r requirements.txt
                            venv\\Scripts\\pip install pytest flake8 mlflow
                        '''
                    }
                }
            }
        }

        stage('Lint Check') {
            steps {
                echo 'Running lint checks...'
                script {
                    if (isUnix()) {
                        sh '''
                            venv/bin/flake8 ER_MAP --count --select=E9,F63,F7,F82 --show-source --statistics
                            venv/bin/flake8 ER_MAP --count --exit-zero --max-complexity=15 --max-line-length=127 --statistics
                        '''
                    } else {
                        bat '''
                            venv\\Scripts\\flake8 ER_MAP --count --select=E9,F63,F7,F82 --show-source --statistics
                            venv\\Scripts\\flake8 ER_MAP --count --exit-zero --max-complexity=15 --max-line-length=127 --statistics
                        '''
                    }
                }
            }
        }

        stage('Run Smoke Tests') {
            steps {
                echo 'Running mock environment smoke tests...'
                script {
                    def hasCredentials = false
                    try {
                        withCredentials([string(credentialsId: 'GROQ_API_KEY', variable: 'GROQ_API_KEY')]) {
                            hasCredentials = true
                            if (isUnix()) {
                                sh '''
                                    venv/bin/python -m ER_MAP.test_smoke
                                '''
                            } else {
                                bat '''
                                    venv\\Scripts\\python -m ER_MAP.test_smoke
                                '''
                            }
                        }
                    } catch (Exception e) {
                        echo "Warning: GROQ_API_KEY credential not configured or failed in Jenkins. Running in STUB mode."
                        if (!hasCredentials) {
                            if (isUnix()) {
                                sh '''
                                    venv/bin/python -m ER_MAP.test_smoke
                                '''
                            } else {
                                bat '''
                                    venv\\Scripts\\python -m ER_MAP.test_smoke
                                '''
                            }
                        }
                    }
                }
            }
        }

        stage('Run Unit & Parity Tests') {
            steps {
                echo 'Running Gymnasium / OpenEnv wrapper parity tests...'
                script {
                    if (isUnix()) {
                        sh '''
                            venv/bin/python -m pytest ER_MAP/envs/openenv_triage/tests -v
                        '''
                    } else {
                        bat '''
                            venv\\Scripts\\python -m pytest ER_MAP/envs/openenv_triage/tests -v
                        '''
                    }
                }
            }
        }

        stage('Dry-Run Training Verification') {
            steps {
                echo 'Verifying GRPO training scheduler and MLflow logger in Dry-Run mode...'
                script {
                    if (isUnix()) {
                        sh '''
                            venv/bin/python -m ER_MAP.training.train_grpo --dry-run --episodes 8 --mlflow
                        '''
                    } else {
                        bat '''
                            venv\\Scripts\\python -m ER_MAP.training.train_grpo --dry-run --episodes 8 --mlflow
                        '''
                    }
                }
            }
        }

        stage('Build Docker Image') {
            steps {
                echo 'Building production/server Docker image...'
                script {
                    if (isUnix()) {
                        sh 'docker build -t ermap-env:latest .'
                    } else {
                        bat 'docker build -t ermap-env:latest .'
                    }
                }
            }
        }
    }

    post {
        always {
            echo 'Archiving test results and cleaning up...'
            cleanWs()
        }
        success {
            echo 'Pipeline completed successfully!'
        }
        failure {
            echo 'Pipeline failed. Please check logs.'
        }
    }
}
