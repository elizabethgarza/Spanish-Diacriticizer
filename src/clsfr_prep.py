import nltk
from typing import Dict




class Clsfr_Prep: 
    
    def __init__(self, train, dev, test): 
        
        self.train = train 
        self.dev = dev
        self.test = test
        
    def train_prep(self): 
        
        train_prep = []
        
        for (sent_info, label) in self.train: 
            train_prep.append((Clsfr_Prep.extract_features(sent_info[0], sent_info[-1]), label))
            
        return train_prep 

    def dev_prep(self):
       
        dev_prep = []
        
        for (sent_info, label) in self.dev: 
            dev_prep.append((Clsfr_Prep.extract_features(sent_info[0], sent_info[-1]), label))
        
        return dev_prep   
    
    
    def test_prep(self):
       
        test_prep = []
        
        for (sent_info, label) in self.test: 
            test_prep.append((Clsfr_Prep.extract_features(sent_info[0], sent_info[-1]), label))
        
        return test_prep   
        
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