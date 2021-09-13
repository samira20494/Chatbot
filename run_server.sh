echo -e "Starting base serving daemon ..."
docker pull tensorflow/serving
docker run -d --name serving_base tensorflow/serving

echo -e "Exporting model as SavedModel object ..."
python3 serving/save_model.py
docker cp model/saved_model serving_base:/models/bert

echo -e "Starting new serving container ..."
docker commit --change "ENV MODEL_NAME bert" serving_base qa_bert
docker kill serving_base
docker run -d -p 8501:8501 -p 8500:8500 --name bert qa_bert
