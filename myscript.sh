#upload all files

#upload body files
gsutil -m cp -r gs://bucket1_314978370/body .

#upload anchor files
gsutil -m cp -r gs://bucket1_314978370/anchor .

#upload title files
gsutil -m cp -r gs://bucket1_314978370/title .

#upload others
gsutil -m cp -r gs://bucket1_314978370/uploads .


#manually upload py files

#install pyspark
pip install pyspark

#run the program
python3 search_frontend.py


# ========================

# upload files
gsutil -m cp -r gs://bucket1_314978370/uploads .
gsutil -m cp -r gs://bucket1_314978370/optimization .
pip install scikit-learn==1.0.2 pandas==1.3.5
#upload train_model.py
#upload new_train.json
python3 train_model.py
