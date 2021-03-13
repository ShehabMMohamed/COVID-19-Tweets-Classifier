import os
import re
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


class Model:
    def __init__(self):
        self.logit_regressor = LogisticRegression(random_state=42)
        self.data_loader = DataLoader()
        self.model = None

    def run(self):
        x_train_vec, y_train, x_val_vec, y_val, x_test_vec, y_test = self.data_loader.run()
        print("Model Training...")
        self.model = self.logit_regressor.fit(x_train_vec, y_train)
        print(classification_report(y_val, self.model.predict(x_val_vec)))


class DataLoader:
    def __init__(self, train_path='Corona_NLP_train.csv', test_path='Corona_NLP_test.csv'):
        self.train_df = pd.read_csv(train_path, encoding='latin-1')
        self.test_df = pd.read_csv(test_path, encoding='latin-1')
        self.stopWord = stopwords.words('english')
        self.covid_corpus = None

    def visualize_labels(self):
        labels = ['Positive', 'Negative', 'Neutral', 'Extremely Positive', 'Extremely Negative']
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff5645']
        explode = (0.05, 0.05, 0.05, 0.05, 0.05)
        plt.pie(self.train_df.Sentiment.value_counts(), colors=colors, labels=labels,
                autopct='%1.1f%%', startangle=90, pctdistance=0.85, explode=explode)
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
        plt.tight_layout()
        plt.show()

    def run(self):
        print("DataLoader running...")
        train, test = self.preprocess()
        x_train = train.Corpus
        y_train = train.Sentiment
        x_test = test.Corpus
        y_test = test.Sentiment

        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
        print("Train Shape...\nX: {}\nY: {}\nValidation Shape...\nX: {}\nY: {}\nTest Shape...\nX: {}\nY: {}\n".format(
            x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_test.shape, y_test.shape))

        vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2), min_df=5).fit(self.covid_corpus)

        x_train_vec = vectorizer.transform(x_train)
        x_val_vec = vectorizer.transform(x_val)
        x_test_vec = vectorizer.transform(x_test)

        return x_train_vec, y_train, x_val_vec, y_val, x_test_vec, y_test

    def preprocess(self):
        self.train_df['Identity'] = 0
        self.test_df['Identity'] = 1
        covid = pd.concat([self.train_df, self.test_df])
        covid.reset_index(drop=True, inplace=True)
        # Shrink 5 classes to 3 classes.
        # Extremely Postive -> Positive
        # Extremely Negative -> Negative
        covid['Sentiment'] = covid['Sentiment'].str.replace('Extremely Positive', 'Positive')
        covid['Sentiment'] = covid['Sentiment'].str.replace('Extremely Negative', 'Negative')

        covid = covid.drop('ScreenName', axis=1)
        covid = covid.drop('UserName', axis=1)

        # labels = ['Positive', 'Negative', 'Neutral']
        # colors = ['lightblue', 'lightsteelblue', 'silver']
        # explode = (0.1, 0.1, 0.1)
        # plt.pie(covid.Sentiment.value_counts(), colors=colors, labels=labels,
        #         shadow=300, autopct='%1.1f%%', startangle=90, explode=explode)
        # plt.show()

        covid['Sentiment'] = covid['Sentiment'].map({'Neutral': 0, 'Positive': 1, 'Negative': 2})
        covid['OriginalTweet'] = covid['OriginalTweet'].apply(lambda x: self.clean(x))

        covid = covid[['OriginalTweet', 'Sentiment', 'Identity']]

        covid['Corpus'] = [nltk.word_tokenize(text) for text in covid.OriginalTweet]
        lemma = nltk.WordNetLemmatizer()
        covid.Corpus = covid.apply(lambda x: [lemma.lemmatize(word) for word in x.Corpus], axis=1)
        covid.Corpus = covid.apply(lambda x: " ".join(x.Corpus), axis=1)

        self.covid_corpus = covid.Corpus

        train = covid[covid.Identity == 0]
        test = covid[covid.Identity == 1]
        train.drop('Identity', axis=1, inplace=True)
        test.drop('Identity', axis=1, inplace=True)
        test.reset_index(drop=True, inplace=True)

        return train, test

    def clean(self, text):
        text = re.sub(r'http\S+', " ", text)
        text = re.sub(r'@\w+', ' ', text)
        text = re.sub(r'#\w+', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        text = re.sub('r<.*?>', ' ', text)
        text = text.split()
        text = " ".join([word for word in text if word not in self.stopWord])

        return text


model = Model()
model.run()
