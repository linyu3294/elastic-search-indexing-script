from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import math


class KLSum:
    vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english')

    def __init__(self):
        nltk.download('punkt')

    def kl_summarize(self, doc_data, num_of_sentences):
        summaries = []
        for document, file_name in doc_data:
            summary = ''
            # skip this iteration summary task if document contains not enough words
            try:
                doc = self.vectorizer.fit_transform([document])
            except ValueError:
                summaries.append('Not Enough Words in Doc')
                continue;

            document_model = self.vectorizer.fit_transform([document])
            summary = []
            picked_sentences = set()
            for _ in range(num_of_sentences):
                this_sentence = ''
                this_sentence_score = float('-inf')
                this_sentence_index = 0
                for sentence in nltk.sent_tokenize(document):
                    # print('sampling sentence: ', sentence, '\n')
                    if this_sentence_index in picked_sentences: continue
                    new_sentences = list(map(lambda x: x[0], summary))
                    new_sentences.append(sentence)
                    kl_score = self.kl_similarity(document_model.T.toarray(),
                                                  self.vectorizer.transform([' '.join(new_sentences)]).T.toarray())
                    if kl_score > this_sentence_score:
                        this_sentence_score = kl_score
                        this_sentence = (sentence, this_sentence_index)
                    this_sentence_index += 1
                if this_sentence != '':
                    summary.append(this_sentence)
                    picked_sentences.add(this_sentence[1])

            summary = sorted(summary, key=lambda x: x[1])
            summaries.append(' '.join(list(map(lambda x: x[0], summary))))
        return summaries

    @staticmethod
    def kl_similarity(p, q):
        kl = 0
        lambda_param = 0.1
        for i in range(p.shape[0]):
            p_i = p[i]
            q_i = q[i]
            kl += p_i * math.log((p_i + lambda_param) / (q_i + (lambda_param * p_i.shape[0])))
        return kl



