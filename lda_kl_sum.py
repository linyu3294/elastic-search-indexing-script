import math
import nltk
from nltk.corpus import stopwords
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


class LdaKlSummary:
    vectorizer = TfidfVectorizer()

    @staticmethod
    def kl_similarity(kl_lambda, p, q, words_list_len):
        kl = p * math.log((p + kl_lambda) / (q + (kl_lambda * words_list_len)))
        return kl

    def make_lda_model(self, topics, max_iter):
        model = LatentDirichletAllocation(n_components=topics,
                                          max_iter=max_iter,
                                          learning_method='online',
                                          learning_offset=1,
                                          random_state=0)
        return model

    def fetch_data_20ng(self):
        ng_train = fetch_20newsgroups(subset='train',
                                      remove=('headers', 'footers', 'quotes')).data
        ng_test = fetch_20newsgroups(subset='test',
                                     remove=('headers', 'footers', 'quotes')).data
        ng_data = ng_train + ng_test
        return ng_data

    def clean_data(self, raw_corpus):
        nltk.download('stopwords')
        clean_corpus = []
        for d in range(len(raw_corpus)):
            temp = raw_corpus[d]
            words_list = temp.split(' ')
            remove_list = stopwords.words('english')
            remove_list.extend(['', '\n', '\n\n', '\n\n\n', '\n\n\n\n'])
            words_list = [i for i in words_list if i not in remove_list]
            doc = ' '.join(words_list)
            clean_corpus.append(doc)
        return clean_corpus

    def get_doc_topic_and_distribution(self,
                                       num_words_in_topic,
                                       clean_corpus,
                                       model):
        lda_topics_distribution = []
        for d in range(len(clean_corpus)):
            doc_topic_distribution = ''
            # skip this iteration summary task if document contains not enough words
            try:
                doc = self.vectorizer.fit_transform([clean_corpus[d]])
            except ValueError:
                lda_topics_distribution.append('none')
                continue;

            result = model.fit(doc)
            topic = result.components_
            feature_names = self.vectorizer.get_feature_names()

            top_words_distribution = list(np.sort(topic[0])[-num_words_in_topic:])
            top_words_indices = list(np.argsort(topic[0])[-num_words_in_topic:])

            for i in range(len(top_words_indices)):
                doc_topic_distribution = doc_topic_distribution + '   ( ' + \
                                         feature_names[top_words_indices[i]] + \
                                         ' : ' + \
                                         str(top_words_distribution[i]) + \
                                         ' )   '
            lda_topics_distribution.append(doc_topic_distribution)
        return lda_topics_distribution

    def get_summaries(self,
                      summary_size,
                      num_words_in_topic,
                      raw_corpus,
                      clean_corpus,
                      model,
                      kl_alpha,
                      kl_lambda):
        vectorizer = TfidfVectorizer()
        summaries = []

        for d in range(len(clean_corpus)):

            summary = ''
            # skip this iteration summary task if document contains not enough words
            try:
                doc = vectorizer.fit_transform([clean_corpus[d]])
            except ValueError:
                summaries.append('Not Enough Words in Doc')
                continue;

            result = model.fit(doc)
            topic = result.components_
            feature_names = vectorizer.get_feature_names()
            top_words_indices = list(np.argsort(topic[0])[-num_words_in_topic:])

            # calculate topic distribution per document
            topic_appear_in_doc_count = 0
            words_list = clean_corpus[d].split(' ')
            topics = []
            for index in top_words_indices:
                topic_appear_in_doc_count += words_list.count(feature_names[index])
                topics.append(feature_names[index])

            true_doc_distr = topic_appear_in_doc_count / len(words_list)

            # print('\n\n\n------------------------------')
            # print('topics : ', ' '.join(topics))
            # print('true topic doc distribution : ', true_doc_distr)
            # # for index in top_words_indices:
            # #     print( '\n\n\n\n topics words for document ', d, ' : ',  feature_names[index])
            # print('total number of sentences : ', len(raw_corpus[d].split('.')))

            # calculate topic distribution per sentence
            topic_appear_in_old_summary_count = 0
            topic_appear_in_new_summary_count = 0
            sentences = raw_corpus[d].split('.')
            sentence_divergence_list = []

            for s in sentences:
                old_eval_list = summary.split(' ')
                new_eval_list = (summary + ' ' + s).split(' ')

                for index in top_words_indices:
                    topic_appear_in_old_summary_count += old_eval_list.count(feature_names[index])
                    topic_appear_in_new_summary_count += new_eval_list.count(feature_names[index])

                summary_old_distr = topic_appear_in_old_summary_count / len(summary.split(' ')) + kl_alpha
                summary_new_distr = topic_appear_in_new_summary_count / len(summary.split(' ')) + kl_alpha
                old_kl_score = self.kl_similarity(kl_lambda, true_doc_distr, summary_old_distr, len(summary.split(' ')))
                new_kl_score = self.kl_similarity(kl_lambda, true_doc_distr, summary_new_distr, len(summary.split(' ')))

                kl_improvement = abs(new_kl_score - old_kl_score)
                sentence_divergence_list.append(kl_improvement)
                evaluated_count = len(sentence_divergence_list)

                avg_divergence_improvement = sum(sentence_divergence_list) / evaluated_count
                # print('sentence : ', evaluated_count, s.strip())
                # print('improvement of kl by adding current sentence: ', kl_improvement, '   | ',
                #       'average kl improvement : ', avg_divergence_improvement)

                # determine if sentence should be included in summary based on kl-divergence scores
                if kl_improvement > avg_divergence_improvement or kl_improvement > true_doc_distr:
                    summary += (s.strip() + '.')
                    # print('====>. this sentence got added to summary')
                if len(summary.split('.')) >= summary_size:
                    break;
            summaries.append(summary)
        return summaries
