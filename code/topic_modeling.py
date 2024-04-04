import pandas as pd
import numpy as np

import re
import argparse

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler as mms
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

from umap import UMAP

config = {
    'embedding_model' : 'all-MiniLM-L6-v2',
    'random_state' : 42,
    
    'topic_cluster_params' : {
        'n_clusters' : 100,
        'random_state' : 42
    },
    
    'topic_umap_params' : {
        'n_neighbors' : 100,
        'n_components' : 10,
        'min_dist' : 0.0,
        'metric' : 'cosine',
        'random_state' : 42
    },

    'group_umap_params' : {
        'n_neighbors' : 2,
        'n_components' : 3,
        'min_dist' : 0.0,
        'metric' : 'hellinger',
        'spread' : 2,
        'random_state' : 42
    },

    'ngram_representation' : {
        'stop_words' : 'english'
    }
}

class TopicModeling():
    """
    Class for handling BERTopic modeling as it pertains to Reddit. 
    """

    def __init__(self, config=config):
        self.config = config

    def load_data_frame(self):
        """
        Load our Reddit post/comments dataframe.
        """
        if self.config['data'].split('.')[-1] != 'xlsx':
            raise TypeError('Expected an Excel file. Other input dataframe types not yet supported.')

        df = pd.read_excel(self.config['data'])
        self.df = self.preprocess_dataframe(df)
        self.texts = self.df['content'].to_list()

    def create_topic_model(self):
        """
        Perform topic modeling analysis.
        """

        self.find_topics()
        self.find_groups()

        # Save this to our dataframe.
        # First, find topic keywords to label topics.
        self.df['topic'] = self.topics
        n_topics = self.df['topic'].nunique()

        topic_kw = ['"' + '", "'.join([i[0] for i in self.topic_model.get_topic(j)]) + '"' for j in range(0, n_topics)]
        self.df['topic_keywords'] = [topic_kw[i] for i in self.df['topic']]

        # Assign groups to each discussion.
        self.df['group'] = [self.groups[i] for i in self.df['topic']]
        
        # Export.
        self.df.to_excel(self.config['output'])

    @staticmethod
    def preprocess_dataframe(df):
        """
        Preprocess dataframe for topic modeling, if this has not already been performed.
        """
        # Fill empty cells and remove some weird html tags
        df['body'].fillna("", inplace=True)
        df['body'] = df['body'].str.replace("http\S+", "")
        df['body'] = df['body'].str.replace("\\n", " ")
        df['body'] = df['body'].str.replace("&gt;", "")
        
        # Get rid of extra spaces
        df['body'] = df['body'].str.replace('\s+', ' ', regex=True)
        
        # Remove those too small.
        df['body_len'] = df['body'].str.len()
        df = df.query('body_len >= 25')
        return df

    def find_initial_num_topics(self):
        """
        Find an initial guess for the # of topics by using HDBScan's output over 3 iterations as a guess.
        """

        n_topics = 0
        n_iter = 3


        vectorizer_model = CountVectorizer(stop_words="english")
        spec_topic_model = BERTopic(vectorizer_model=vectorizer_model)

        print('==== Finding an initial guess for the # of topics ====')
        for i in range(n_iter):
            topics, _ = spec_topic_model.fit_transform(self.texts, self.embeddings)
            print(f'Number of topics in run {i} : {np.max(topics)}')
            n_topics += (np.max(topics) + 1) / n_iter

        return int(n_topics)

    def find_topics(self):
        """
        Creates the topic model as specified below.
        """
        
        ### Create our initial model classes.
        # Create our sentence model which will the level of our tokenization.
        self.embedding_model = SentenceTransformer(self.config['embedding_model'])

        # Create our vectorizer model to improve the keyword representation of our topics.
        self.vectorizer_model = CountVectorizer(**self.config['ngram_representation'])

        # Create our specific UMAP and clustering model. 
        self.umap_model = UMAP(**self.config['topic_umap_params'])
        self.cluster_model = SpectralClustering(**self.config['cluster_params'])

        # Calculate our embeddings to permit reuse.
        self.embeddings = self.embedding_model.encode(self.texts, show_progress_bar=True)

        # Now, we must find an initial guess for the optimal # of topics. 
        n_topics = self.find_initial_num_topics()

        # Now, we can instantiate our clustering model with this guess.
        cluster_model = SpectralClustering(n_clusters=n_topics, random_state=self.config['random_state'])

        # Create our topic model.
        self.topic_model = BERTopic(umap_model=self.umap_model, hdbscan_model=self.cluster_model, vectorizer_model=self.vectorizer_model)
        self.topics, _ = self.topic_model.fit_transform(self.texts, self.embeddings)

    def find_groups(self):
        # Normalize the ctfidf representation of topics.
        c_tf_idf_mms = mms().fit_transform(self.topic_model.c_tf_idf.toarray())
        
        # This helps us to visualize
        # self.c_tf_idf_vis = UMAP(n_neighbors=2, n_components=2, metric='hellinger', self.config['random_state']).fit_transform(c_tf_idf_mms)
        self.c_tf_idf_embed = UMAP(**self.config['group_umap_params']).fit_transform(c_tf_idf_mms)
        
        # Find the ideal # of groups.
        ideal_n_clusters = self.find_ideal_num_groups(self.c_tf_idf_embed)
        self.groups = SpectralClustering(n_clusters=ideal_n_clusters, random_state=self.config['random_state']).fit_predict(self.c_tf_idf_embed) + 1

    def find_ideal_num_groups(self, c_tf_idf_embed, llim=3, ulim=25, return_plot_data=False):
        """
        Find the optimal number of clusters based on Silhouette score and Davies-Bouldin score across a range of groups.
        """
        ss = []
        db = []

        cluster_arr = np.arange(llim, ulim)
        
        for n_clusters in cluster_arr:
            clusters = SpectralClustering(n_clusters=n_clusters, random_state=self.config['random_state']).fit_predict(c_tf_idf_embed)
            ss.append(silhouette_score(c_tf_idf_embed, clusters))
            db.append(davies_bouldin_score(c_tf_idf_embed, clusters))

        print("top silhouette score: {0:0.3f} for at n_clusters {1}".format(np.max(ss), ideal_n_clusters))
        print("top davies-bouldin score: {0:0.3f} for at n_clusters {1}".format(np.min(db), cluster_arr[np.argmin(db)]))

        ideal_n_clusters = cluster_arr[np.argmax(ss)]           
        
        return ideal_n_clusters

    def save_topic_model(self):
        # Saves the current topic model.
        raise NotImplementedError

    def load_topic_model(self, path_to_topic_model):
        # Load a saved topic model.
        raise NotImplementedError

if __name__ == '__main__':
    
    # Enter input data via argparse.
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type='str', default='data/cac_db.xlsx', help='Path to dataset')
    parser.add_argument('output', type='str', default='data/topic_model_res.xlsx', help='Path to save topic, group labels')
    args = parser.parse_args()

    config['data'] = args.data
    config['output'] = args.output

    topic_model = TopicModeling(config)
    topic_model.load_data_frame()
    topic_model.create_topic_model()