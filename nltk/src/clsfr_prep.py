import nltk
import sentences
from typing import Dict




class Clsfr_Prep: 
    
    def __init__(self, train, test): 
        
        self.train = train 
        self.test = test
        
    def mk_nltk_train_set(self): 
        
        nltk_train_set = []
        
        for (sentence_info, label) in self.train: 
            nltk_train_set.append((Clsfr_Prep.extract_features(sentence_info[0], sentence_info[-1]), label))
            
        return nltk_train_set 
    
    
    def mk_nltk_test_set(self):
       
        nltk_test_set = []
        
        for (sentence_info, label) in self.test: 
            nltk_test_set.append((Clsfr_Prep.extract_features(sentence_info[0], sentence_info[-1]), label))
        
        return nltk_test_set   
        
    @staticmethod
    def extract_features(tokenized_sentence, index: int):
        
        features = {} 

        features["is initial token"] = (index == 0)
        if index == 0: 
            if len(tokenized_sentence)==1: 
                features["is only token"] = (len(tokenized_sentence)==1)
            if len(tokenized_sentence)==2:
                features["one token after:"] = tokenized_sentence[index+1]
            if len(tokenized_sentence)>3: 
                features["one token after:"] = tokenized_sentence[index+1]
                features["two tokens after:"] = tokenized_sentence[index+2] 
        if index==1: 
            features["is 2nd token"] = (index == 1)
            if len(tokenized_sentence)==2: 
                features["one token before:"] = tokenized_sentence[index-1]
            if len(tokenized_sentence)==3: 
                features["one token before:"] = tokenized_sentence[index-1]
                features["one token after:"] = tokenized_sentence[index+1]
            if len(tokenized_sentence)>3: 
                features["one token before:"] = tokenized_sentence[index-1] 
                features["one token after:"] = tokenized_sentence[index+1]
                features["two tokens after:"] = tokenized_sentence[index+2] 
        if index>1: 
            if index==2:
                if len(tokenized_sentence) == 3:
                    features["one token before:"] = tokenized_sentence[index-1]
                    features["two tokens before:"] = tokenized_sentence[index-2]
                    features["is 3rd token"] = (index==2)
                if len(tokenized_sentence)==4: 
                    features["one token before:"] = tokenized_sentence[index-1]
                    features["two tokens before:"] = tokenized_sentence[index-2]
                    features["one token after:"] = tokenized_sentence[index+1]
            if index<(len(tokenized_sentence)-2): 
                features["one token before:"] = tokenized_sentence[index-1]
                features["two tokens before:"] = tokenized_sentence[index-2]
                features["one token after:"] = tokenized_sentence[index+1]
                features["two tokens after:"] = tokenized_sentence[index+2] 
            if index==(len(tokenized_sentence)-2): 
                features["one token before:"] = tokenized_sentence[index-1]
                features["two tokens before:"] = tokenized_sentence[index-2]
                features["one token after:"] = tokenized_sentence[index+1]
                features["is penultimate token"] = (index == len(tokenized_sentence)-2)
            if index==(len(tokenized_sentence)-1):
                features["one token before:"] = tokenized_sentence[index-1]
                features["two tokens before:"] = tokenized_sentence[index-2]
        features["is last token"] = (index == len(tokenized_sentence)-1)     

        return features