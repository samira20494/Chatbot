cd app/
# Start rasa server with nlu model
echo PORT $PORT
rasa run -m models --enable-api --cors "*" --debug --endpoints heroku-endpoints.yml -p $PORT
