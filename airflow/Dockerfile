FROM apache/airflow:2.10.2-python3.11

USER root

# Install system dependencies (Debian 12 Bookworm repos work out-of-the-box)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git gcc python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /mlflow/artifacts && \
    chown -R 50000:0 /mlflow && \
    chmod -R 777 /mlflow

USER airflow

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY dags/ /opt/airflow/dags/
COPY config/ /opt/airflow/config/

ENV AIRFLOW__CORE__LOAD_EXAMPLES=False