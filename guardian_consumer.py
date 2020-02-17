# -*- coding: utf-8 -*-

# system libs
import string
import re
from time import sleep
import json, sys
import requests
import time
from string import punctuation
from unidecode import unidecode
# external libs
import langid
import nltk
from kafka import KafkaConsumer
from pyspark.sql import SparkSession, SQLContext
from pyspark import SparkContext, SparkConf
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, IntegerType, FloatType
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.feature import HashingTF, Tokenizer, IDF
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from elasticsearch import Elasticsearch


nltk.download('stopwords')
nltk.download('wordnet')
cachedStopWords = stopwords.words("english")
wordnet_lemmatizer = WordNetLemmatizer()

def check_lang(data_str):
    predict_lang = langid.classify(data_str)
    language = predict_lang[0]
    return language

def remove_features(data_str):
    # remove special chars
    data_str = unidecode(data_str)
    data_str = data_str.lower()
    remove_num = ''.join(letter for letter in data_str if not letter.isdigit())
    remove_punc = ''.join(letter if letter not in punctuation else ' ' for letter in remove_num)
    return remove_punc

def remove_stops(data_str):
    # data_str.replace('"', '').replace("'", "")
    removed_text = ' '.join([word for word in data_str.split() if word not in cachedStopWords])
    return removed_text

def lemmatize(data_str):
    removed_text = ' '.join([wordnet_lemmatizer.lemmatize(word) for word in data_str.split()])
    return removed_text

def text_preprocess(data_str):
    rm_feature_text = remove_features(data_str)
    rm_stops_text = remove_stops(rm_feature_text)
    final_text = lemmatize(rm_stops_text)
    return final_text

def check_accur(label, prediction):
    if label == prediction:
        return 1.0
    else:
        return 0.0

if __name__== "__main__":
    # create spark session
    conf = SparkConf().setMaster("local").setAppName("hw3_model_training")
    sc = SparkContext(conf=conf)
    spark = SparkSession(sc)

    # setup kafka consumer
    consumer = KafkaConsumer('guardian2', bootstrap_servers=['localhost:9092'])

    # # create RDD from collected text array
    # data_rdd = spark.sparkContext.parallelize(all_news)
    # # map RDD to seq as text, index, label
    # data_rdd = data_rdd.map(lambda x: x.split('||')).zipWithUniqueId().map(lambda x: (x[0][1], x[1], int(x[0][0])))
    # # convert RDD to spark DataFrame
    # data_df = spark.createDataFrame(data_rdd).toDF('text', 'idx', 'label')
    # data_df.show(5)
    # # check text language and just consider the english text
    # check_lang_udf = udf(check_lang, StringType())
    # lang_df = data_df.withColumn('lang', check_lang_udf(data_df['text']))
    # en_df = lang_df.filter(lang_df['lang'] == 'en')
    # # en_df.select('text').show(1, truncate=False)

    # text_preprocess_udf = udf(text_preprocess, StringType())
    # cleaned_text_df = en_df.withColumn('cleaned_text', text_preprocess_udf(en_df['text']))
    # test_data_df = cleaned_text_df.select(cleaned_text_df['idx'], cleaned_text_df['cleaned_text'], cleaned_text_df['label'])
    # test_data_df.show(5)

    es = Elasticsearch(['localhost'], port=9200)

    # LR model
    same_model = PipelineModel.load("./model/lr")
    # NB model
    # same_model = PipelineModel.load("./model/nb")
    # Random Forest model
    # same_model = PipelineModel.load("./model/rf")

    score = 0
    total = 0
    es.indices.delete(index='guardian', ignore=[400, 404])

    for news in consumer:
        # print(news.value.decode('UTF-8'))
        # create RDD from collected text array
        data = ''.join(letter for letter in news.value.decode('UTF-8'))
        data_rdd = spark.sparkContext.parallelize([data])
        # map RDD to seq as text, index, label
        data_rdd = data_rdd.map(lambda x: x.split('||')).zipWithUniqueId().map(lambda x: (x[0][1], x[1], int(x[0][0])))
        # test = spark.createDataFrame(data_rdd).toDF('text', 'idx', 'label')
        # test.show()
        # convert RDD to spark DataFrame
        data_df = spark.createDataFrame(data_rdd).toDF('text', 'idx', 'label')
        data_df.show(5)
        # check text language and just consider the english text
        check_lang_udf = udf(check_lang, StringType())
        lang_df = data_df.withColumn('lang', check_lang_udf(data_df['text']))
        en_df = lang_df.filter(lang_df['lang'] == 'en')
        # en_df.select('text').show(1, truncate=False)

        text_preprocess_udf = udf(text_preprocess, StringType())
        cleaned_text_df = en_df.withColumn('cleaned_text', text_preprocess_udf(en_df['text']))
        test_data_df = cleaned_text_df.select(cleaned_text_df['cleaned_text'], cleaned_text_df['label'])
        test_data_df.show(5)
        pr = same_model.transform(test_data_df)
        selected = pr.select('cleaned_text', 'label', 'probability', 'prediction')
        check_accur_udf = udf(check_accur, FloatType())
        selected = selected.withColumn('score', check_accur_udf(selected['label'], selected['prediction']))
        # res = selected.filter(selected.label == selected.prediction)
        # print(res.count())
        # res.show()
        accur = selected.filter(selected.label == selected.prediction)
        selected.show()
        total += 1
        if accur.count() == 1:
            score += 1
        res = selected.toJSON().map(lambda j: json.loads(j)).collect()

        for i in res:
            es.index(index='guardian', doc_type='text', body=i)
        print('Accuracy: ' + str(score / total))

    sc.stop()