import os
from nltk import word_tokenize
from tqdm import tqdm


class Preprocess: 
    
    def __init__(self, corpus, melliza1, melliza2): 
        
        self.corpus = corpus 
        self.melliza1 = melliza1 
        self.melliza2 = melliza2 
        
    def find_sentences_for_melliza1(self): 

        os.chdir('/Users/mariagarza/Desktop/nltk-Spanish-diacriticizer/data')
        with open(self.corpus, 'r') as source: 
            sentence_set = set()
            lines = []
            i=0
            for line in tqdm(source):
                tokenized_line = line.split()
                for token in tokenized_line: 
                    lowercased_token = token.casefold()
                    if self.melliza1 == lowercased_token:
                        lines.append(' '.join(tokenized_line))
            for line in lines: 
                sentence_set.add(line)

            return list(sentence_set)
        
    def find_sentences_for_melliza2(self): 
        
        with open(self.corpus, 'r') as source: 
            sentence_set = set()
            lines = []
            for line in tqdm(source):
                tokenized_line = line.split()
                for token in tokenized_line: 
                    lowercased_token = token.casefold()
                    if self.melliza2 == lowercased_token:
                        lines.append(' '.join(tokenized_line))
            for line in lines: 
                sentence_set.add(line)

            return list(sentence_set)
        
    @staticmethod
    def get_sentence_info_for_feature_extraction(sentence, melliza_of_interest): 
    
        tokenized_sentence = word_tokenize(sentence)
        melliza_of_interest_index = -1
        for token in tokenized_sentence:
            casefolded_token = token.casefold()
            melliza_of_interest_index +=1
            if casefolded_token == melliza_of_interest: 
                break

        return tokenized_sentence, melliza_of_interest_index

    
    def mk_melliza1_melliza2_list(self): 
        
        melliza1_sentence_info_list = []
        for sentence in tqdm(Preprocess.find_sentences_for_melliza1(self)): 
            melliza1_sentence_info_list.append((Preprocess.get_sentence_info_for_feature_extraction(sentence, self.melliza1), self.melliza1))
        
        melliza2_sentence_info_list = []
        for sentence in tqdm(Preprocess.find_sentences_for_melliza2(self)): 
            melliza2_sentence_info_list.append((Preprocess.get_sentence_info_for_feature_extraction(sentence, self.melliza2), self.melliza2))
            
        melliza1_melliza2_info_list = melliza1_sentence_info_list + melliza2_sentence_info_list
        
        return melliza1_melliza2_info_list 

    def mk_train_and_test_set(self): 
        
        melliza1_melliza2_sentences_and_labels = Preprocess.mk_melliza1_melliza2_list(self)
        #random.seed(5228554)
        #random.shuffle(melliza1_melliza2_sentences_and_labels)
        train_length = round(.8 * len(melliza1_melliza2_sentences_and_labels))    
        train = melliza1_melliza2_sentences_and_labels[:train_length]
        test = melliza1_melliza2_sentences_and_labels[train_length:]
        
        return train, test

        
