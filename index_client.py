import nltk
from nltk.corpus import stopwords
from elasticsearch import Elasticsearch
from elasticsearch import helpers
from kl_sum import KLSum
from lda_kl_sum import LdaKlSummary
from news_reader import NewsgroupParser
from duc_reader import DocumentParser
from rouge_score import rouge_scorer


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
                            kl_summary,
                            lda_kl_score,
                            kl_score,
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
                    'lda_kl_summary': lda_kl_summary.get(key),
                    'lda_kl_score': lda_kl_score.get(key),
                    'kl_summary': kl_summary.get(key),
                    'kl_score': kl_score.get(key),
                }
            }
            actions.append(action)
        return actions

    @staticmethod
    def prepare_ng_actions(document,
                           lda_topics,
                           lda_kl_summary,
                           kl_summary,
                           index_name) -> list:
        actions = []
        for key, value in document.items():
            action = {
                '_index': index_name,
                '_id': key,
                '_source': {
                    'docno': key,
                    'content': document.get(key),
                    'lda_topics': lda_topics.get(key),
                    'lda_kl_summary': lda_kl_summary.get(key),
                    'kl_summary': kl_summary.get(key)
                }
            }
            actions.append(action)
        return actions

    def load_duc_docs(self,
                      document,
                      gold_summary,
                      lda_topics_distr_dict,
                      lda_kl_summary,
                      kl_summary,
                      lda_kl_score,
                      kl_score,
                      index_name):
        if self.esclient.indices.exists(index_name):
            actions = self.prepare_duc_actions(document,
                                               gold_summary,
                                               lda_topics_distr_dict,
                                               lda_kl_summary,
                                               kl_summary,
                                               lda_kl_score,
                                               kl_score,
                                               index_name)
            success, failures = helpers.bulk(client=self.esclient, actions=actions)
            print("Uploading DUC dataset ... Please wait")
            print("Successfully Loaded : ", str(success), ' documents.')
            print("Failed to Load : ", str(failures), ' documents.')
        else:
            print('index does not exist.')

    def load_ng_docs(self, document, duc_lda_topics_distr_dict, lda_kl_summary, kl_summary, index_name):
        if self.esclient.indices.exists(index_name):
            actions = self.prepare_ng_actions(document,
                                              duc_lda_topics_distr_dict,
                                              lda_kl_summary,
                                              kl_summary,
                                              index_name)
            success, failures = helpers.bulk(client=self.esclient, actions=actions)
            print("Uploading 20 News Group dataset ... Please wait")
            print("Successfully Loaded : ", str(success), ' documents.')
            print("Failed to Load : ", str(failures), ' documents.')
        else:
            print('index does not exist.')


# create indexed entries for each document in the two datasets, duc and 20ng
# upload analysis and training results to elastic search
# For both techniques, kl-sum (words density) and kl-sum-lda-topic-modeliing,
# create entries in elastic search that are searchable by tag in kibana discover

if __name__ == '__main__':
    esclient = IndexClient()
    #
    # in the duc dataset, there are 308 distinct documents
    # 'docno' - assign unique key to each document
    # 'gold summary' - specific to duc entries, summaries written by a actual person
    # entries that belong to 20 newsgroup do not have 'gold summary' field
    # content - the actual content in the document
    # 'lda_topics' - a list of top words most indicative of a topic for a document (determined by a LDA topic model)
    # 'lda_topics' - also displays probability distribution of top topic words
    # 'kl_summary' - summary created based on words_PD;
    #                Probaility distribution is a distribution proportional to counts of words in document
    # 'lda_kl_summary' - summary based on LDA topics_PD
    #                    surveys sentences and minimizes kl divergence
    #                    of true distribution and summary distribution given sentences.
    # 'rogue_score' -tbd

    esclient.create_index('duc')
    duc_parser = DocumentParser('./DUC2001/dataset/', './DUC2001/summaries/')
    duc_document_dict, duc_gold_summary_dict = duc_parser.map()

    lda_kl_sum = LdaKlSummary()
    # sentences per summary
    duc_summary_size = 5
    # pick top n probalistic words per topic
    duc_num_words_in_topic = 10
    duc_kl_alpha = 100
    duc_kl_lambda = 0.00001
    duc_model = lda_kl_sum.make_lda_model(topics=1, max_iter=10)
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
                                                    duc_kl_lambda
                                                    )

    duc_lda_kl_summary_dict = dict(zip(list(duc_document_dict.keys()), duc_lda_kl_summaries))

    duc_kl_sum = KLSum()
    duc_kl_summaries = duc_kl_sum.kl_summarize(zip(duc_data_raw, duc_data_raw), 5)
    duc_kl_summary_dict = dict(zip(list(duc_document_dict.keys()), duc_kl_summaries))

    duc_gold_summaries = list(duc_gold_summary_dict.values())
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    print('\n\nTesting rouge scores : ')
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score('The quick brown fox jumps over the lazy dog',
                          'The quick brown dog jumps on the log.')
    print(scores)

    print('\n\nDUC KL summaries rouge scores : ')
    duc_kl_sum_rouge_score = []
    for s in range(len(duc_gold_summaries)):
        score = scorer.score(str(duc_gold_summaries[s]),
                             str(duc_kl_summaries[s]))
        duc_kl_sum_rouge_score.append(score)
        print(scores)
    duc_kl_sum_rouge_score_dict = dict(zip(list(duc_document_dict.keys()), duc_kl_sum_rouge_score))

    print('\n\nDUC LDA KL summaries rouge scores : ')
    duc_lda_kl_sum_rouge_score = []
    for s in range(len(duc_gold_summaries)):
        score = scorer.score(str(duc_gold_summaries[s]),
                             str(duc_lda_kl_summaries[s]))
        duc_lda_kl_sum_rouge_score.append(score)
        print(scores)
    duc_lda_kl_sum_rouge_score_dict = dict(zip(list(duc_document_dict.keys()), duc_lda_kl_sum_rouge_score))

    esclient.load_duc_docs(duc_document_dict,
                           duc_gold_summary_dict,
                           duc_lda_topics_distr_dict,
                           duc_lda_kl_summary_dict,
                           duc_kl_summary_dict,
                           duc_lda_kl_sum_rouge_score_dict,
                           duc_kl_sum_rouge_score_dict,
                           index_name='duc')

    #
    # esclient.create_index('20ng')
    # news_group_parser = NewsgroupParser()
    # ng_document_dict = news_group_parser.map()
    #
    # lda_kl_sum = LdaKlSummary()
    # # sentences per summary
    # ng_summary_size = 5
    # # pick top n probalistic words per topic
    # ng_num_words_in_topic = 20
    # ng_kl_alpha = 10
    # ng_kl_lambda = 0.00001
    # ng_model = lda_kl_sum.make_lda_model(topics=1, max_iter=10)
    # ng_data_raw = list(ng_document_dict.values())
    # ng_data_clean = lda_kl_sum.clean_data(ng_data_raw)
    #
    # ng_lda_topics_distribution = lda_kl_sum.get_doc_topic_and_distribution(ng_num_words_in_topic,
    #                                                                        ng_data_clean,
    #                                                                        ng_model)
    #
    # ng_lda_topics_distr_dict = dict(zip(list(ng_document_dict.keys()), ng_lda_topics_distribution))
    #
    # ng_lda_kl_summaries = lda_kl_sum.get_summaries(ng_summary_size,
    #                                                ng_num_words_in_topic,
    #                                                ng_data_raw,
    #                                                ng_data_clean,
    #                                                ng_model,
    #                                                ng_kl_alpha,
    #                                                ng_kl_lambda
    #                                                )
    #
    # ng_lda_kl_summary_dict = dict(zip(list(ng_document_dict.keys()), ng_lda_kl_summaries))
    #
    # ng_kl_sum = KLSum()
    # ng_kl_summaries = ng_kl_sum.kl_summarize(zip(ng_data_raw, ng_data_raw), 5)
    # ng_kl_summary_dict = dict(zip(list(ng_document_dict.keys()), ng_kl_summaries))
    #
    # esclient.load_ng_docs(ng_document_dict,
    #                       ng_lda_topics_distr_dict,
    #                       ng_lda_kl_summary_dict,
    #                       ng_kl_summary_dict,
    #                       index_name='20ng')
