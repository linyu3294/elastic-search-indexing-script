from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import LatentDirichletAllocation
from news_reader import NewsgroupParser


class lda_model:

    def __init__(self):
        return None

    @staticmethod
    def create_model(k_topics=10, max_iter=5):
        l = LatentDirichletAllocation(n_components=k_topics,
                                      max_iter=max_iter,
                                      learning_method='online',
                                      learning_offset=1,
                                      random_state=0)
        return l

    # print(len(doc_top_distr))
    # print(len(lda.components_[2]))

    def get_top_words_per_topic(self, index_name, model, vocabulary, n_top_words):
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
        return topic_items


if __name__ == '__main__':
    VECTORIZER = CountVectorizer()
    news = NewsgroupParser()
    data, _ = news.fetch_data_20ng()
    _, ng_raw = news.fetch_data_20ng()

    ng_data_vect = VECTORIZER.fit_transform(data)

    model = lda_model()

    k_topics = 10
    max_iter = 5
    lda = model.create_model(k_topics, max_iter)
    doc_top_distr = lda.fit_transform(ng_data_vect)

    for doc_id, doc_topic_distribution in zip(ng_raw.filenames, doc_top_distr):
        top_topics = doc_top_distr.argsort()[::1][:max_iter]

    feature_names = VECTORIZER.get_feature_names()
    topic_items = model.get_top_words_per_topic("lda_ng_topics", lda, feature_names, 10)
