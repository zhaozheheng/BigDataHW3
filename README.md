# BigDataHW3

The purpose of Homework 3 is to implement a system for real-time news stream classification, trend analysis and viasualization. For such implementation, you will work with Apache Spark and Apache Kafka, implement some of the state-of-the-art classifiers like SVM by using MLlib, and utilize Spark Streaming.
You are asked to extract features and build a classifier over a stream of news articles. For the task of gathering real-time news articles, you will use stream tools provided by Guardian API. This part of the implementation is provided for you in the file stream_producer.py. Note that, for using this API, you are required to register/sign up for an API key in this link:
https://open-platform.theguardian.com/access/
Stream_producer.py generates the Kafka streaming data from Guardian API every 1 second. The topic created has been named 'guardian2'. When running this from your command prompt window, you should be able to see the news getting printed on the screen in the following format:
Label index (news category) || headline + bodyText
Each news article will be tagged with one category, such as Australia news, US news, Football, World news, Sport, Television & radio, Environment, Science, Media, News, Opinion, Politics, Business, UK news, Society, Life and style, Inequality, Art and design, Books, Stage, Film, Music, Global, Food, Culture, Community, Money, Technology, Travel, From the Observer, Fashion, Crosswords, Law, etc. Stream_producer.py script can capture a category for each news article and assign a label index for that category.
Example of Stream_producer.py usage:
$ python3 stream_producer.py API-key fromDate toDate For instance:
python3 stream_producer.py API-key 2018-11-3 2018-12-24
  
To simulate real-time streaming and processing you will collect as much data as you can by streamming through kafka (by using ‘Stream_producer.py’). The classification model will be built offline with articles from such collected data.
Following, in Spark context, you will need to create a Pipeline model with Tokenizer, Stopword remover, Labelizer, TF-IDF vectorizer, and your Classifier. You are allowed to test many different classification techniques and later present performance results (i.e., Accuracy, Recall, etc.) for the chosen classification technique.
Finally, you will visualize the results over sliding Windows using Kibana. You need to use Spark Streaming for performing classification and visualize the output using ElasticSearch and Kibana. For example, a frequency count task can be performed using ElasticSearch whose results can be visualized in a tag cloud. As the stream goes on, the framework processes news articles continuously and periodically updates the tag cloud.
The following figure depicts the framework structure.
 Note that, for this homework, you are required to set up standalone Spark and Kafka on your own system. Besides, you are allowed to use NLP processing tools, python Scikit-learn and MLlib in Spark packages.
Additional Notes:
(i) When applying/testing your classification model, you are required to use different window streamming (‘fromDate’ and ‘toDate’) than the one used for training the classification model.
(ii) The label (categories of the news) must be used only for the training process (always as dependent variable). Furthermore, during testing, the labels can be utilized to report accuracy, recall, and precision for each batch.
