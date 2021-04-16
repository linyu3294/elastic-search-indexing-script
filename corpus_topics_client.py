from elasticsearch.helpers import bulk
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import LatentDirichletAllocation
from news_reader import NewsgroupParser
from duc_reader import DocumentParser
from elasticsearch import Elasticsearch
from nltk.corpus import stopwords


class CorpusTopicsClient:

    def __init__(self):
        return None

    @staticmethod
    def create_model(k_topics=200, max_iter=5):
        l = LatentDirichletAllocation(n_components=k_topics,
                                      max_iter=max_iter,
                                      learning_method='online',
                                      learning_offset=1,
                                      random_state=0)
        return l

    @staticmethod
    def get_top_words_per_topic(index_name, model, vocabulary, n_top_words):
        topic_items = []
        for topic_id, topic_words_prob in enumerate(model.components_):
            print("Topic #%d: " % topic_id)
            words = []
            for i in topic_words_prob.argsort()[:-n_top_words - 1: -1]:
                word = {'word': vocabulary[i], 'probability': topic_words_prob[i]}
                words.append(word)
            topic_item = {
                '_type': 'type1',
                '_index': index_name,
                '_source': {
                    'topic_id': topic_id,
                    'topic_words': words
                }
            }

            topic_items.append(topic_item)
            print(' '.join(vocabulary[i] for i in topic_words_prob.argsort()[:-n_top_words - 1: -1]))

        es = Elasticsearch()
        # es.indices.delete(index= index_name, ignore=[400, 404])
        print('Bulk Loading Topics')
        success, failures = bulk(es, topic_items)
        print("Successfully Loaded : ", str(success), ' documents.')
        print("Failed to Load : ", str(failures), ' documents.')
        es.indices.refresh(index=index_name)
        return topic_items


if __name__ == '__main__':

    model = CorpusTopicsClient()
    lda = model.create_model(k_topics=20, max_iter=5)
    VECTORIZER = CountVectorizer()

    news = NewsgroupParser()
    ng_data, _ = news.fetch_data_20ng()

    for i in range(len(ng_data)):
        temp = ng_data[i]
        words_list = temp.split(' ')
        remove_list = stopwords.words('english')
        remove_list.extend([''])
        words_list = [i for i in words_list if i not in remove_list]
        doc = ' '.join(words_list)
        ng_data[i] = doc

    _, ng_raw = news.fetch_data_20ng()
    ng_data_vect = VECTORIZER.fit_transform(ng_data)
    doc_topic_distr = lda.fit_transform(ng_data_vect)
    for doc_id, doc_topic_distribution in zip(ng_raw.filenames, doc_topic_distr):
        top_topics = doc_topic_distr.argsort()[::1][:20]
    feature_names = VECTORIZER.get_feature_names()
    topic_items = model.get_top_words_per_topic("20ng-topics", lda, feature_names, 20)

    duc = DocumentParser()
    duc_dict, _ = duc.map()
    duc_file_names = duc_dict.keys()
    duc_data = list(duc_dict.values())
    remove_list = stopwords.words('english')
    remove_list.extend([''])

    for i in range(len(duc_data)):
        temp = duc_data[i]
        words_list = temp.split(' ')
        remove_list = stopwords.words('english')
        remove_list.extend([''])
        words_list = [i for i in words_list if i not in remove_list]
        doc = ' '.join(words_list)
        duc_data[i] = doc

    ng_data_vect = VECTORIZER.fit_transform(duc_data)
    doc_topic_distr = lda.fit_transform(ng_data_vect)
    for doc_id, doc_topic_distribution in zip(duc_file_names, doc_topic_distr):
        top_topics = doc_topic_distr.argsort()[::1][:20]
    feature_names = VECTORIZER.get_feature_names()
    topic_items = model.get_top_words_per_topic("duc-topics", lda, feature_names, 20)
