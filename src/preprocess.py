import os
from nltk import word_tokenize
from tqdm import tqdm

"""
TODOs:  
 1.  merge mk_clsfr_dict with this, but use the top_600_mellizas 
 2.  possible change the title of mk_clsfr_dict to mk_clsfrs
 3.  run on a baby file to test, then do everything.
 """

class Preprocess: 
    
    def __init__(self, corpus, token1, token2): 
        
        self.corpus = corpus 
        self.token1 = token1 
        self.token2 = token2 
        
    def find_sentences_for_token1(self): 

        os.chdir('/Users/mariagarza/Desktop/nltk-Spanish-diacriticizer/data')
        with open(self.corpus, "r") as source: 
            sentence_set = set()
            lines = []
            i=0
            for line in tqdm(source):
                tokenized_line = line.split()
                for token in tokenized_line: 
                    lowercased_token = token.casefold()
                    if self.token1 == lowercased_token:
                        lines.append(" ".join(tokenized_line))
            for line in lines: 
                sentence_set.add(line)

            return list(sentence_set)
        
    def find_sentences_for_token2(self): 
        
        with open(self.corpus, "r") as source: 
            sentence_set = set()
            lines = []
            i=0
            for line in tqdm(source):
                tokenized_line = line.split()
                for token in tokenized_line: 
                    lowercased_token = token.casefold()
                    if self.token2 == lowercased_token:
                        lines.append(" ".join(tokenized_line))
            for line in lines: 
                sentence_set.add(line)

            return list(sentence_set)
        
    @staticmethod
    def get_sentence_info_for_feature_extraction(sentence, token_of_interest): 
    
        tokenized_sentence = word_tokenize(sentence)
        token_of_interest_index = -1
        for item in tokenized_sentence:
            casefolded_token = item.casefold()
            token_of_interest_index +=1
            if casefolded_token == token_of_interest: 
                break

        return tokenized_sentence, token_of_interest_index

    
    def mk_token1_token2_list(self): 
        
        token1_sentence_info_list = []
        for sentence in tqdm(Preprocess.find_sentences_for_token1(self)): 
            token1_sentence_info_list.append((Preprocess.get_sentence_info_for_feature_extraction(sentence, self.token1), self.token1))
        
        token2_sentence_info_list = []
        for sentence in tqdm(Preprocess.find_sentences_for_token2(self)): 
            token2_sentence_info_list.append((Preprocess.get_sentence_info_for_feature_extraction(sentence, self.token2), self.token2))
            
        token1_token2_info_list = token1_sentence_info_list + token2_sentence_info_list
        
        return token1_token2_info_list 

    def mk_train_and_test_set(self): 
        
        token1_token2_sentences_and_labels = Preprocess.mk_token1_token2_list(self)
        #random.seed(5228554)
        #random.shuffle(token1_token2_sentences_and_labels)
        train_length = round(.8 * len(token1_token2_sentences_and_labels))    
        train = token1_token2_sentences_and_labels[:train_length]
        test = token1_token2_sentences_and_labels[train_length:]
        
        return train, test

        
