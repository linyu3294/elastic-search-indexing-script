from rouge_score import rouge_scorer

from duc_reader import DocumentParser
from kl_sum import KLSum
from lda_kl_sum import LdaKlSummary

if __name__ == '__main__':

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
    for s in range(10):

        scores = scorer.score(str(duc_gold_summaries[s]),
                              str(duc_kl_summaries[s]))
        print(scores)

    print('\n\nDUC LDA KL summaries rouge scores : ')
    for s in range(10):
        scores = scorer.score(str(duc_gold_summaries[s]),
                              str(duc_lda_kl_summaries[s]))
        print(scores)

