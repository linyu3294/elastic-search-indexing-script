import os
from bs4 import BeautifulSoup


class DocumentParser:
    list_of_files = []
    dataset_path = ''
    summaries_path = ''

    def __init__(self, dataset_path='./DUC2001/dataset/', summaries_path='./DUC2001/summaries/'):
        self.dataset_path = dataset_path
        self.summaries_path = summaries_path
        self.list_of_files = os.listdir(dataset_path)

        if 'notes.txt' in self.list_of_files:
            self.list_of_files.remove('notes.txt')
        if 'annotations' in self.list_of_files:
            self.list_of_files.remove('annotations')
        return None

    @staticmethod
    def read_file(file_path) -> str:
        file = open(file_path, 'r')
        corpus = file.read()
        file.close()
        return corpus

    def map(self) -> list:
        document_dict = {}
        summary_dict = {}
        for file_name in self.list_of_files:
            file_path = self.dataset_path + file_name
            with open(file_path) as fp:
                soup = BeautifulSoup(fp, 'html.parser')
                doc_id = soup.find('doc').find('docno').getText().strip()
                doc_text = soup.find('doc').find('text').getText().strip()
                doc_summary_path = self.summaries_path + file_name.lower() + '.txt'
                try:
                    with open(doc_summary_path) as sp:
                        summary = sp.read()
                except FileNotFoundError:
                    summary = 'Empty Introduction'
                document_dict[doc_id] = doc_text
                summary_dict[doc_id] = summary.split('Introduction')[0]
        return [document_dict, summary_dict]


if __name__ == '__main__':
    doc_parser = DocumentParser('./DUC2001/dataset/', './DUC2001/summaries/')
    document_dict, summary_dict = doc_parser.map()
    available_num_summary = 0;

    for key, val in summary_dict.items():
        if val != -1:
            available_num_summary += 1
    print('total summaries :  ', available_num_summary)
    print('total documents :  ', len(document_dict.keys()))
    # for key, val in content_dict.items():
    #     print('Document ID is ' , key , '\n', val + '\n\n\n\n')
