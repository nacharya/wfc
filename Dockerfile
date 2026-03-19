FROM python:3.14-slim

LABEL wfc=true

# System packages: build tools needed for numpy/scikit-learn wheels on slim image
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        vim bash git net-tools build-essential && \
    rm -rf /var/lib/apt/lists/*

ARG user=app
ARG group=app
ARG uid=1001
ARG gid=1001

RUN groupadd --gid ${gid} ${group} && \
    useradd  --uid ${uid} --gid ${gid} --shell /bin/bash --create-home ${user}

WORKDIR /home/app

COPY bootup.sh          /home/app/bootup.sh
COPY requirements-docker.txt /home/app/requirements.txt

RUN pip install --upgrade pip --no-cache-dir && \
    pip install --no-cache-dir -r /home/app/requirements.txt && \
    python -c "import nltk; nltk.download('vader_lexicon', quiet=True)"

RUN mkdir -p /app
COPY app/ /app

RUN chown -R ${user}:${group} /app && \
    mkdir -p /home/app/.config/matplotlib && \
    chmod -R 777 /home/app/.config/matplotlib

RUN mkdir -p /app/.streamlit
COPY dotstreamlit/secrets.toml      /app/.streamlit/
COPY dotstreamlit/credentials.toml  /app/.streamlit/
COPY dotstreamlit/config.toml       /app/.streamlit/

USER ${user}

EXPOSE 9900

WORKDIR /app

CMD ["/home/app/bootup.sh"]
