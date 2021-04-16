import nltk
from nltk.corpus import stopwords
from elasticsearch import Elasticsearch
from elasticsearch import helpers

from lda_kl_sum import LdaKlSummary
from news_reader import NewsgroupParser
from duc_reader import DocumentParser


class IndexClient:
    esclient = None
    settings = {}
    mappings = {}

    @staticmethod
    def download_stop_words():
        nltk.download('stopwords')

    def __init__(self):
        self.esclient = Elasticsearch(["localhost:9200"])
        self.download_stop_words()
        self.mappings = {
            'properties': {
                'text': {
                    'type': 'text',
                    'fielddata': 'true',
                    'analyzer': 'stopped'
                },
                'gold_summary': {
                    'type': 'text',
                    'fielddata': 'true',
                    'analyzer': 'stopped'
                }
            }
        }
        self.settings = {
            'number_of_shards': 1,
            'number_of_replicas': 1,
            'analysis': {
                'filter': {
                    'english_stop': {
                        'type': 'stop',
                        'stopwords': stopwords.words('english')
                    },
                    'stemmer': {
                        'type': 'snowball',
                        'language': 'English'
                    }
                },
                'analyzer': {
                    'stopped': {
                        'type': 'custom',
                        'tokenizer': 'standard',
                        'filter': [
                            'lowercase',
                            'english_stop',
                            'stemmer'
                        ]
                    }
                }
            }

        }

    def create_index(self, index_name):
        self.esclient.indices.create(index=index_name,
                                     body={'settings': self.settings, 'mappings': self.mappings},
                                     ignore=400)

    @staticmethod
    def prepare_duc_actions(document,
                            gold_summary,
                            lda_topics,
                            lda_kl_summary,
                            index_name) -> list:
        actions = []
        for key, values in document.items():
            action = {
                '_index': index_name,
                '_id': key,
                '_source': {
                    'docno': key,
                    'content': document.get(key),
                    'gold_summary': gold_summary.get(key),
                    'lda_topics': lda_topics.get(key),
                    'lda_kl_summary': lda_kl_summary.get(key)
                }
            }
            actions.append(action)
        return actions

    @staticmethod
    def prepare_ng_actions(document, index_name) -> list:
        actions = []
        for key, value in document.items():
            action = {
                '_index': index_name,
                '_id': key,
                '_source': {
                    'docno': key,
                    'content': value
                }
            }
            actions.append(action)
        return actions

    def load_duc_docs(self, document, gold_summary, duc_lda_topics_distr_dict, lda_kl_summary, index_name):
        if self.esclient.indices.exists(index_name):
            actions = self.prepare_duc_actions(document,
                                               gold_summary,
                                               duc_lda_topics_distr_dict,
                                               lda_kl_summary,
                                               index_name)
            success, failures = helpers.bulk(client=self.esclient, actions=actions)
            print("Successfully Loaded : ", str(success), ' documents.')
            print("Failed to Load : ", str(failures), ' documents.')
        else:
            print('index does not exist.')

    def load_ng_docs(self, document, index_name):
        if self.esclient.indices.exists(index_name):
            actions = self.prepare_ng_actions(document, index_name)
            helpers.bulk(client=self.esclient, actions=actions)
        else:
            print('index does not exist.')


# create indexed entries for each document in the two datasets, duc and 20ng
# upload analysis and training results to elastic search
# For both techniques, kl-sum (words density) and kl-sum-lda-topic-modeliing,
# create entries in elastic search that are searchable by tag in kibana discover

if __name__ == '__main__':
    esclient = IndexClient()

    # in the duc dataset, there are 308 distinct documents
    # 'docno' - assign unique key to each document
    # 'gold summary' - specific to duc entries, summaries written by a actual person
    # entries that belong to 20 newsgroup do not have 'gold summary' field
    # content - the actual content in the document
    # 'top_topics' - a list of top words most indicative of a topic for a document (determined by a LDA topic model)
    # 'top_topics_pd' - probability distribution of top topic words
    # 'kl_summary' - summary created based on words_PD;
    #                Probaility distribution is a distribution proportional to counts of words in document
    # 'lda_kl_summary' - summary based on LDA topics_PD
    #                    surveys sentences and minimizes kl divergence
    #                    of true distribution and summary distribution given sentences.
    # 'rogue_score' -tbd

    lda_kl_sum = LdaKlSummary()

    esclient.create_index('duc')
    duc_parser = DocumentParser('./DUC2001/dataset/', './DUC2001/summaries/')
    duc_document_dict, duc_gold_summary_dict = duc_parser.map()

    # sentences per summary
    duc_summary_size = 5
    # pick top n probalistic words per topic
    duc_num_words_in_topic = 10
    duc_kl_alpha = 100
    duc_kl_lambda = 0.00001
    duc_model = lda_kl_sum.make_lda_model(topics=1, max_iter=10);
    duc_data_raw = list(duc_document_dict.values())
    duc_data_clean = lda_kl_sum.clean_data(duc_data_raw)

    duc_lda_topics_distribution = lda_kl_sum.get_doc_topic_and_distribution(duc_num_words_in_topic,
                                                                            duc_data_clean,
                                                                            duc_model)

    duc_lda_topics_distr_dict = dict(zip(list(duc_document_dict.keys()), duc_lda_topics_distribution))

    duc_lda_kl_summaries = lda_kl_sum.get_summaries(duc_summary_size,
                                                    duc_num_words_in_topic,
                                                    duc_data_raw,
                                                    duc_data_clean,
                                                    duc_model,
                                                    duc_kl_alpha,
                                                    duc_kl_lambda)

    duc_lda_kl_summary_dict = dict(zip(list(duc_document_dict.keys()), duc_lda_kl_summaries))

    esclient.load_duc_docs(duc_document_dict,
                           duc_gold_summary_dict,
                           duc_lda_topics_distr_dict,
                           duc_lda_kl_summary_dict,
                           index_name='duc')

    # esclient.create_index('20ng')
    # news_group_parser = NewsgroupParser()
    # ng_document_dict = news_group_parser.map()
    # print( len( ng_document_dict ) )
    # esclient.load_ng_docs(ng_document_dict, index_name='20ng')
