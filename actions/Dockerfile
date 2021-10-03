FROM rasa/rasa-sdk:2.8.2

COPY . /app/actions/
COPY ./startup.sh /startup.sh
COPY requirements.txt .
COPY COVID-QA.json .

EXPOSE 5005
ENV PORT 5005


STOPSIGNAL SIGINT

USER root
RUN apt-get update && apt-get install -y python3 python3-pip && python3 -m pip install --no-cache --upgrade pip
RUN pip3 install -r requirements.txt
RUN chmod a+rx /startup.sh
ENTRYPOINT "/startup.sh"