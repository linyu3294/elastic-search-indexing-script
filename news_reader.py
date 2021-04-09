from sklearn.datasets import fetch_20newsgroups


class NewsgroupParser:
    ng_data = []

    def fetch_data_20ng(self):
        ng_train = fetch_20newsgroups(subset='train',
                                      remove=('headers', 'footers', 'quotes')).data
        ng_test = fetch_20newsgroups(subset='test',
                                     remove=('headers', 'footers', 'quotes')).data
        ng_data = ng_train + ng_test
        ng_raw = fetch_20newsgroups()
        return ng_data, ng_raw

    def __init__(self):
        newsgroups_train = fetch_20newsgroups(subset='train').data
        newsgroups_test = fetch_20newsgroups(subset='test').data
        self.ng_data = newsgroups_train + newsgroups_test
        return None

    def map(self) -> list:
        document_dict = {}
        for i in range(len(self.ng_data)):
            lines = self.ng_data[i].split('\n\n')
            doc = self.ng_data[i].replace(lines[0], '')
            document_dict[i] = doc
        return document_dict


if __name__ == '__main__':
    news_group_parser = NewsgroupParser()
    news_group_parser.map()
