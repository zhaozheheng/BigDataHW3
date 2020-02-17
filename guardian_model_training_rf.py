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
from pyspark.sql.types import StringType
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.feature import HashingTF, Tokenizer, IDF
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

nltk.download('stopwords')
nltk.download('wordnet')
cachedStopWords = stopwords.words("english")
wordnet_lemmatizer = WordNetLemmatizer()

def getData(url):
    data = []
    labels = {}
    index = 0

    curr_page = 1
    jsonData = requests.get(url).json()
    total_pages = jsonData["response"]['pages']
    print(total_pages)
    while curr_page <= total_pages:
        jsonData = requests.get(url + '&page=' + str(curr_page)).json()
        # print(url + '&page=' + str(curr_page))
        # print(jsonData["response"]['pages'])
        for i in range(len(jsonData["response"]['results'])):
            headline = jsonData["response"]['results'][i]['fields']['headline']
            bodyText = jsonData["response"]['results'][i]['fields']['bodyText']
            headline += bodyText
            label = jsonData["response"]['results'][i]['sectionName']
            if label not in labels:
                labels[label] = index
                index += 1
            #data.append({'label':labels[label],'Descript':headline})
            toAdd=str(labels[label])+'||'+headline
            data.append(toAdd)
        curr_page += 1
    json.dump(labels, open("./data/labels.json", 'w'))
    print(labels)
    return(data)

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

if __name__== "__main__":
    API_key = '679fa611-5736-42bf-8d4e-e0f468ba481c'
    # setup kafka consumer
    # consumer = KafkaConsumer('guardian2', bootstrap_servers=['localhost:9092'])

    # create spark session
    conf = SparkConf().setMaster("local").setAppName("hw3_model_training")
    sc = SparkContext(conf=conf)
    spark = SparkSession(sc)

    # Create a schema for the dataframe
    # cSchema = StructType([
    #     StructField('text', StringType(), True),
    #     # StructField('Count', IntegerType(), True),
    #     # StructField('Description', StringType(), True)
    # ])

    # for message in consumer:
    #     print(message)
    url = 'http://content.guardianapis.com/search?from-date=' + '2018-10-3' + '&to-date=' + '2018-11-3' + '&order-by=newest&show-fields=all&page-size=200&%20num_per_section=10000&api-key=' + API_key
    all_news=getData(url)
    # print(all_news)
    # index = range(len(all_news))
    print(url)
    print('datasize: ' + str(len(all_news)))
    # create RDD from collected text array
    data_rdd = spark.sparkContext.parallelize(all_news)
    # map RDD to seq as text, index, label
    data_rdd = data_rdd.map(lambda x: x.split('||')).zipWithUniqueId().map(lambda x: (x[0][1], x[1], int(x[0][0])))
    # convert RDD to spark DataFrame
    data_df = spark.createDataFrame(data_rdd).toDF('text', 'idx', 'label')
    data_df.show(5)
    # check text language and just consider the english text
    check_lang_udf = udf(check_lang, StringType())
    lang_df = data_df.withColumn('lang', check_lang_udf(data_df['text']))
    en_df = lang_df.filter(lang_df['lang'] == 'en')
    # en_df.select('text').show(1, truncate=False)

    # convert to lowercase, remove misc junk like URLs, Punctuation, Numbers, @Mentions
    # remove_features_udf = udf(remove_features, StringType())
    # rm_features_df = en_df.withColumn('feat_text', remove_features_udf(en_df['text']))
    # rm_features_df.select('feat_text').show(1, truncate=False)

    # remove stop words from the text
    # remove_stops_udf = udf(remove_stops, StringType())
    # rm_stops_df = rm_features_df.withColumn('stop_text', remove_stops_udf(rm_features_df['feat_text']))
    # rm_stops_df.select('stop_text').show(1, truncate=False)

    # lemmatize words in the text
    # lemmatize_udf = udf(lemmatize, StringType())
    # lemmatize_df = rm_stops_df.withColumn('lemmatized_text', lemmatize_udf(rm_stops_df['stop_text']))
    # lemmatize_df.select('lemmatized_text').show(1, truncate=False)

    text_preprocess_udf = udf(text_preprocess, StringType())
    cleaned_text_df = en_df.withColumn('cleaned_text', text_preprocess_udf(en_df['text']))
    # cleaned_text_df.select('cleaned_text').show(1, truncate=False)
    # train_pre_data_df = cleaned_text_df.select(cleaned_text_df['cleaned_text'], cleaned_text_df['text_label'], cleaned_text_df['idx'])
    train_pre_data_df = cleaned_text_df.select(cleaned_text_df['idx'], cleaned_text_df['cleaned_text'], cleaned_text_df['label'])
    train_pre_data_df.show(5)

    train_data, test_data = train_pre_data_df.randomSplit([0.9, 0.1], seed=12345)

    tokenizer = Tokenizer(inputCol='cleaned_text', outputCol='words')
    hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol='features')
    idf = IDF(minDocFreq=3, inputCol='features', outputCol='idf')
    rf = RandomForestClassifier(numTrees=3, maxDepth=5)
    pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, rf])

    paramGrid = ParamGridBuilder().addGrid(hashingTF.numFeatures, [10, 100, 1000]).build()

    crossval = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=BinaryClassificationEvaluator(), numFolds=3)

    cvModel = crossval.fit(train_pre_data_df)
    pip_model = cvModel.bestModel
    # cvModel.save("~/model/lr")
    # cvModel = crossval.fit(train_data)
    # pr = cvModel.transform(test_data)
    # selected = pr.select('idx', 'cleaned_text', 'label', 'probability', 'prediction')
    pip_model.save("./model/rf")
    # selected.show()
    sc.stop()