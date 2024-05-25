FROM python:3.11

# Label so we can later obtain this stage from the multi-stage build to get the coverage results
LABEL wfc=true

RUN apt update && apt upgrade -y
RUN apt install -y vim bash git net-tools

ARG user=app
ARG group=app
ARG uid=1001
ARG gid=1001
# create group and user
RUN addgroup --gid ${gid} ${group}
RUN adduser --uid ${uid} --gid ${gid} --shell /bin/bash ${user}
RUN usermod -a -G ${group} ${user}


WORKDIR /home/app

COPY bootup.sh /home/app/bootup.sh

COPY requirements.txt /home/app/requirements.txt 
RUN pip install --upgrade pip
RUN pip install -r requirements.txt 

RUN mkdir -p /app
COPY app/ /app


# assign owners and flags
RUN chown -R ${user}:${group} /app
RUN mkdir -p /home/app/.config/matplotlib
RUN chmod -R 777 /home/app/.config/matplotlib


# Get OpenAI API key 
RUN mkdir -p /app/.streamlit
COPY dotstreamlit/secrets.toml /app/.streamlit
COPY dotstreamlit/credentials.toml /app/.streamlit
COPY dotstreamlit/config.toml /app/.streamlit

USER ${user}


#API
EXPOSE 9910
# xdnview
EXPOSE 9600

WORKDIR /app

CMD [ "/home/app/bootup.sh" ]

