from sklearn.datasets import fetch_20newsgroups


class NewsgroupParser:
    ng_data = []
    ng_raw = []

    def __init__(self):
        ng_train = fetch_20newsgroups(subset='train',
                                      remove=('headers', 'footers', 'quotes')).data
        ng_test = fetch_20newsgroups(subset='test',
                                     remove=('headers', 'footers', 'quotes')).data
        self.ng_data = ng_train + ng_test
        self.ng_raw = fetch_20newsgroups()
        return None

    def fetch_data_20ng(self):
        return self.ng_data, self.ng_raw

    def map(self) -> list:
        document_dict = {}
        for i in range(len(self.ng_data)):
            document_dict[i] = self.ng_data[i]
        return document_dict



if __name__ == '__main__':
    news_group_parser = NewsgroupParser()
    news_group_parser.map()
