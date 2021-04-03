import nltk
from nltk.corpus import stopwords
from elasticsearch import Elasticsearch
from elasticsearch import helpers

from news_reader import NewsgroupParser

from duc_reader import DocumentParser


class ESClient:
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
    def prepare_duc_actions(document, summary, index_name) -> list:
        actions = []
        for key, values in document.items():
            action = {
                '_index': index_name,
                '_id': key,
                '_source': {
                    'docno': key,
                    'content': document.get(key),
                    'gold_summary': summary.get(key)
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

    def load_duc_docs(self, document, summary, index_name):
        if self.esclient.indices.exists(index_name):
            actions = self.prepare_duc_actions(document, summary, index_name)
            helpers.bulk(client=self.esclient, actions=actions)
        else:
            print('index does not exist.')

    def load_ng_docs(self, document, index_name):
        if self.esclient.indices.exists(index_name):
            actions = self.prepare_ng_actions(document, index_name)
            helpers.bulk(client=self.esclient, actions=actions)
        else:
            print('index does not exist.')


if __name__ == '__main__':
    esclient = ESClient()

    esclient.create_index('duc')
    duc_parser = DocumentParser('./DUC2001/dataset/', './DUC2001/summaries/')
    duc_document_dict, duc_summary_dict = duc_parser.map()
    available_num_summary = 0;
    for key, val in duc_summary_dict.items():
        if val != -1:
            available_num_summary += 1
    esclient.load_duc_docs(duc_document_dict, duc_summary_dict, index_name='duc')
    print('total summaries :  ', available_num_summary)
    print('total documents :  ', len(duc_document_dict.keys()))

    # esclient.create_index('20ng')
    # news_group_parser = NewsgroupParser()
    # ng_document_dict = news_group_parser.map()
    # print(len(ng_document_dict))
    # esclient.load_ng_docs(ng_document_dict, index_name='20ng')
