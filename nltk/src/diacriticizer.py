import pandas
from tqdm import tqdm
import unidecode 
import pickle
import json
import nltk

class Diacriticizer: 
         
    
    def __init__(self):
        
        self.melliza_and_clsfr_dict = {}
        counts_of_mellizas = pandas.read_csv("top_200_mellizas.csv", sep=",")
        new_counts = counts_of_mellizas.head(1)
        max1_max2_tokens = new_counts["MAX1_TOKEN"] + "\t"+ new_counts["MAX2_TOKEN"] 
        for row in tqdm(max1_max2_tokens): 
            tokenized_row = row.split()
            read_clsfr = open(f"{unidecode.unidecode(tokenized_row[0])}.pickle", "rb") 
            DECODED_TKN_clsfr = pickle.load(read_clsfr)
            self.melliza_and_clsfr_dict[unidecode.unidecode(tokenized_row[0])] =  DECODED_TKN_clsfr  
            
        with open("invariantly_diacriticized_tokens_dict.json", "r") as source: 
            self.invariantly_diacriticized_tokens_dict = json.load(source)
    
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
                    features["two tokens before:"] = tokenized_sentence[index-2], 
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


    def predict_token(self, tokenized_sentence, index, decoded_lowercased_token):
        
        self.tokenized_sentence = tokenized_sentence 
        self.index = index
        self.decoded_lowercased_token = decoded_lowercased_token
        
        sentence_features = Diacriticizer.extract_features(self.tokenized_sentence, self.index)
        predicted_token = self.melliza_and_clsfr_dict[self.decoded_lowercased_token].classify(sentence_features)
        
        return predicted_token
          
    
    def predict_sentence(self, tokenized_sentence): 
        
        self.tokenized_sentence = tokenized_sentence 

        predicted_tokens = []
        for (index,token) in enumerate(self.tokenized_sentence): 
            decoded_lowercased_token = unidecode.unidecode(token.casefold())
            if decoded_lowercased_token in self.melliza_and_clsfr_dict:
                predicted_tokens.append(self.predict_token(self.tokenized_sentence, index, decoded_lowercased_token))
            elif decoded_lowercased_token in self.invariantly_diacriticized_tokens_dict: 
                predicted_tokens.append(self.invariantly_diacriticized_tokens_dict[decoded_lowercased_token])
            else: 
                predicted_tokens.append(decoded_lowercased_token)
            
        return predicted_tokens

print(Diacriticizer().predict_sentence(nltk.word_tokenize("donde esta el libro.")))