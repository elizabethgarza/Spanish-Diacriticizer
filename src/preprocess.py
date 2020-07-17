from nltk import word_tokenize
import os
import pandas as pd
import subprocess
from tqdm import tqdm
import unidecode
import random


class Preprocess: 
    
    def __init__(self, corpus_path, melliza1, melliza2): 
        
        self.corpus_path = corpus_path 
        self.melliza1 = melliza1 
        self.melliza2 = melliza2
        
        os.chdir('/Users/mariagarza/Desktop/nltk-Spanish-diacriticizer/data')

        with open('top_200_mellizas.csv', 'r') as source: 
            self.mellizas = pd.read_csv('top_200_mellizas.csv', sep=',')
            self.mellizas = self.mellizas.drop(['Unnamed: 0'], axis=1)
            self.mellizas = self.mellizas.set_index('DECODED_MELLIZA')

    def no_of_sents(self):

        melliza_row_value = unidecode.unidecode(self.melliza1).upper()
        melliza_info = self.mellizas.loc[melliza_row_value, ['MAX1_COUNT', 'MAX2_COUNT']]
        MAX1_COUNT = melliza_info.get('MAX1_COUNT')
        MAX2_COUNT = melliza_info.get('MAX2_COUNT')
        if MAX1_COUNT < MAX2_COUNT: 
            no_of_sents = MAX1_COUNT 
        else: 
            no_of_sents = MAX2_COUNT 
            
        return no_of_sents
        
    def melliza1_sents(self): 
         
        max_no_of_sents = Preprocess.no_of_sents(self)
        sent_set = set()
        lines = []
        with open(self.corpus_path, 'r') as source: 
            for line in tqdm(source):
                tokenized_line = line.split()
                for token in tokenized_line: 
                    lowercased_token = token.casefold()
                    if self.melliza1 == lowercased_token:
                        lines.append(' '.join(tokenized_line))
            i=0
            for line in lines:
                i+=1
                if i<max_no_of_sents:
                    sent_set.add(line)
                else: 
                    continue

        return list(sent_set)
        
    def melliza2_sents(self): 
        
        max_no_of_sents = Preprocess.no_of_sents(self)
        sent_set = set()
        lines = []
        with open(self.corpus_path, 'r') as source: 
            for line in tqdm(source):
                tokenized_line = line.split()
                for token in tokenized_line: 
                    lowercased_token = token.casefold()
                    if self.melliza2 == lowercased_token:
                        lines.append(' '.join(tokenized_line))
            i=0
            for line in lines:
                i+=1
                if i<max_no_of_sents:
                    sent_set.add(line)
                else: 
                    continue

        return list(sent_set)

    @staticmethod
    def feat_ex_prep(sent, melliza): 
    
        tokenized_sent = word_tokenize(sent)
        melliza_index = -1
        for token in tokenized_sent:
            casefolded_token = token.casefold()
            melliza_index +=1
            if casefolded_token == melliza: 
                break

        return tokenized_sent, melliza_index

    
    def melliza1_melliza2_list(self): 
        
        melliza1_sent_info = []
        for sent in tqdm(Preprocess.melliza1_sents(self)): 
            melliza1_sent_info.append((Preprocess.feat_ex_prep(sent, self.melliza1), self.melliza1))
        
        melliza2_sent_info = []
        for sent in tqdm(Preprocess.melliza2_sents(self)): 
            melliza2_sent_info.append((Preprocess.feat_ex_prep(sent, self.melliza2), self.melliza2))
            
        melliza1_melliza2_sent_info = melliza1_sent_info + melliza2_sent_info
        
        return melliza1_melliza2_sent_info 
    
    def train_dev_test(self): 
        
        melliza1_melliza2_sents_and_labels = Preprocess.melliza1_melliza2_list(self)
        random.seed(5228554)
        random.shuffle(melliza1_melliza2_sents_and_labels)
        train_length = round(.8 * len(melliza1_melliza2_sents_and_labels))   
        train = melliza1_melliza2_sents_and_labels[:train_length]
        dev_and_test = melliza1_melliza2_sents_and_labels[train_length:]
        print(len(dev_and_test))
        # computes the middle index of dev_and_test
        dev_and_test_len = round(len(dev_and_test) * .5)
        dev = melliza1_melliza2_sents_and_labels[:dev_and_test_len]
        test = melliza1_melliza2_sents_and_labels[dev_and_test_len:]
        
        return train, dev, test



        
