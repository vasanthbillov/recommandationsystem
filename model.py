from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import pickle
import pandas as pd
import numpy as np
import re
import string
import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download('omw-1.4')


class SentimentRecommenderModel:
    ROOT_PATH = "pickles/"
    MODEL_NAME = "xg-boost-model.pkl"
    VECTORIZER = "tfidf-vectorizer.pkl"
    RECOMMENDER = "user_final_rating.pkl"
    CLEANED_DATA = "cleaned-data.pkl"

    def __init__(self):
        self.model = pickle.load(open(
            SentimentRecommenderModel.ROOT_PATH + SentimentRecommenderModel.MODEL_NAME, 'rb'))
        self.vectorizer = pd.read_pickle(
            SentimentRecommenderModel.ROOT_PATH + SentimentRecommenderModel.VECTORIZER)
        self.user_final_rating = pickle.load(open(
            SentimentRecommenderModel.ROOT_PATH + SentimentRecommenderModel.RECOMMENDER, 'rb'))
        self.data = pd.read_csv("data/sample30.csv")
        self.cleaned_data = pickle.load(open(
            SentimentRecommenderModel.ROOT_PATH + SentimentRecommenderModel.CLEANED_DATA, 'rb'))

    def getRecommendationByUser(self, user):
        return list(self.user_final_rating.loc[user].sort_values(ascending=False)[0:20].index)

    def getSentimentRecommendations(self, user):
        if (user in self.user_final_rating.index):
            # get the product recommedation using the trained ML model
            recommendations = list(self.user_final_rating.loc[user].sort_values(ascending=False)[0:20].index)
            filtered_data = self.cleaned_data[self.cleaned_data.id.isin(recommendations)]
            X = self.vectorizer.transform(filtered_data["reviews_text_pos_NN"].values.astype(str))
            filtered_data["predicted_sentiment"] = self.model.predict(X)
            temp = filtered_data[['id', 'predicted_sentiment']]
            temp_grouped = temp.groupby('id', as_index=False).count()
            temp_grouped["pos_review_count"] = temp_grouped.id.apply(lambda x: temp[(
                temp.id == x) & (temp.predicted_sentiment == 1)]["predicted_sentiment"].count())
            temp_grouped["total_review_count"] = temp_grouped['predicted_sentiment']
            temp_grouped['pos_sentiment_percent'] = np.round(
                temp_grouped["pos_review_count"]/temp_grouped["total_review_count"]*100, 2)
            sorted_products = temp_grouped.sort_values(
                'pos_sentiment_percent', ascending=False)[0:5]
            res =  pd.merge(self.data, sorted_products, on="id")[["name", "brand", "manufacturer", "pos_sentiment_percent"]].drop_duplicates().sort_values(['pos_sentiment_percent', 'name'], ascending=[False, True])
            df_to_dict = res.to_dict('list')

            return df_to_dict

        else:
            print(f"User name {user} doesn't exist")
            return None


    


if __name__ == '__main__':
    sentiment_model = SentimentRecommenderModel()
    res = sentiment_model.getSentimentRecommendations('gh')
    print(res['name'][0])


