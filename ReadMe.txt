System requirement:
{
    MacOS
    Python3
    Spark
    Guardian API
    Kafka
    ElasticSearch
    Kibana
}

Since training is a time-consuming task,
I fixed all the command line parameters in python file for convience.

To run my code, just call: python3 file.py

training files:
{
    guardian_model_training_lr.py for LogisticRegression
    guardian_model_training_nb.py for Naive Bayes
    guardian_model_training_rf.py for Random Forest
}

Kafka communication files:
{
    stream_producer.py for Producer side
    guardian_consumer.py for Consumer side
}