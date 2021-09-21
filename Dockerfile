FROM ubuntu:20.04
ENTRYPOINT []
COPY requirements.txt .

# inform which port will run on
EXPOSE 5005

RUN apt-get update && apt-get install -y python3 python3-pip && python3 -m pip install --no-cache --upgrade pip
RUN pip3 install -r requirements.txt
ADD . /app/
RUN chmod +x /app/start_services.sh
CMD /app/start_services.sh

