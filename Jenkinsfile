// Pipeline de integracion continua del proyecto MLOps de riesgo de credito.
// Se configura en Jenkins como "Pipeline script from SCM" apuntando a este repositorio.
pipeline {
    agent any

    parameters {
        string(name: 'CORREO_NOTIFICACION', defaultValue: '',
               description: 'Correo que recibe el resultado (requiere SMTP configurado en Jenkins)')
    }

    triggers {
        // Revisa el repositorio cada 5 minutos: un push o un merge a la rama configurada dispara el pipeline
        pollSCM('H/5 * * * *')
    }

    options {
        timeout(time: 30, unit: 'MINUTES')
        buildDiscarder(logRotator(numToKeepStr: '10'))
    }

    stages {
        stage('Clonar repositorio') {
            steps {
                checkout scm
                sh 'git log -1 --pretty=format:"Commit %h de %an: %s"'
            }
        }

        stage('Validar estructura de carpetas') {
            steps {
                sh '''
                    faltantes=0
                    for ruta in \
                        mlops_pipeline/Cargar_datos.ipynb \
                        mlops_pipeline/comprension_eda.ipynb \
                        mlops_pipeline/ft_engineering.py \
                        mlops_pipeline/model_training_evaluation.py \
                        mlops_pipeline/model_deploy.py \
                        mlops_pipeline/model_monitoring.py \
                        Base_de_datos.csv \
                        requirements.txt \
                        .gitignore \
                        setup.bat \
                        readme.md \
                        Dockerfile \
                        .dockerignore
                    do
                        if [ -e "$ruta" ]; then
                            echo "OK     $ruta"
                        else
                            echo "FALTA  $ruta"
                            faltantes=$((faltantes + 1))
                        fi
                    done
                    if [ "$faltantes" -gt 0 ]; then
                        echo "La estructura esta incompleta: faltan $faltantes archivo(s)."
                        exit 1
                    fi
                    echo "La estructura de carpetas es correcta."
                '''
            }
        }

        stage('Preparar entorno Python') {
            steps {
                sh '''
                    if ! command -v python3 > /dev/null || ! python3 -m venv --help > /dev/null 2>&1; then
                        apt-get update && apt-get install -y python3 python3-venv python3-pip
                    fi
                    python3 -m venv .venv
                    . .venv/bin/activate
                    pip install --quiet --upgrade pip
                    pip install --quiet -r requirements.txt
                '''
            }
        }

        stage('Pruebas unitarias') {
            steps {
                sh '''
                    . .venv/bin/activate
                    python -m pytest --junitxml=reporte-pruebas.xml --cov=mlops_pipeline --cov-report=term
                '''
            }
            post {
                always {
                    junit allowEmptyResults: true, testResults: 'reporte-pruebas.xml'
                }
            }
        }

        stage('Construir imagen Docker') {
            when {
                expression { return sh(script: 'command -v docker > /dev/null', returnStatus: true) == 0 }
            }
            steps {
                sh 'docker build -t mlops-creditos-api:${BUILD_NUMBER} -t mlops-creditos-api:latest .'
            }
        }
    }

    post {
        success {
            script { notificar('EXITOSO') }
        }
        failure {
            script { notificar('FALLIDO') }
        }
    }
}

// Notificacion al terminar: mensaje en la consola, archivo con el resumen y correo opcional.
def notificar(String estado) {
    def mensaje = "Pipeline ${env.JOB_NAME} #${env.BUILD_NUMBER}: ${estado}. Detalle: ${env.BUILD_URL}"
    echo "NOTIFICACION -> ${mensaje}"
    writeFile file: 'notificacion.txt', text: mensaje + '\n'
    archiveArtifacts artifacts: 'notificacion.txt', allowEmptyArchive: true
    if (params.CORREO_NOTIFICACION?.trim()) {
        try {
            mail to: params.CORREO_NOTIFICACION,
                 subject: "[Jenkins] ${env.JOB_NAME} #${env.BUILD_NUMBER}: ${estado}",
                 body: mensaje
        } catch (Exception error) {
            echo "No se pudo enviar el correo (revise la configuracion SMTP de Jenkins): ${error.message}"
        }
    }
}
