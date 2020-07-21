#!/usr/bin/env python3
'''Diacriticizes a sequence of Spanish tokens.'''

import argparse
import json
import nltk
import os
import pandas
import pickle
import unidecode 


class Diacriticize:
    
    def __init__(self):
        
        self.melliza_and_clsfr_dict = {}
        os.chdir('..')
        os.chdir('data') 
        mellizas = pandas.read_csv('top_200_mellizas.csv', sep=',')
        mellizas = mellizas.head(199)
        max1_max2_tokens = mellizas['MAX1_MELLIZA'] + '\t'+ mellizas['MAX2_MELLIZA']
        i=0 
        for row in max1_max2_tokens: 
            tokenized_row = row.split()
            melliza1 = tokenized_row[0]
            os.chdir('pickles')
            read_clsfr = open(f'{unidecode.unidecode(melliza1)}.pickle', 'rb') 
            melliza_clsfr = pickle.load(read_clsfr)
            self.melliza_and_clsfr_dict[unidecode.unidecode(melliza1)] =  melliza_clsfr
            os.chdir('..')
                 
        with open('invars.json', 'r') as source: 
            self.invariantly_diacriticized_tokens_dict = json.load(source)
    
    @staticmethod
    def extract_features(tokenized_sentence, index: int):

        features = {} 

        features['is initial token'] = (index == 0)
        if index == 0: 
            if len(tokenized_sentence)==1: 
                features['is only token'] = (len(tokenized_sentence)==1)
            if len(tokenized_sentence)==2:
                features['one token after:'] = tokenized_sentence[index+1]
            if len(tokenized_sentence)>3: 
                features['one token after:'] = tokenized_sentence[index+1]
                features['two tokens after:'] = tokenized_sentence[index+2] 
        if index==1: 
            features['is 2nd token'] = (index == 1)
            if len(tokenized_sentence)==2: 
                features['one token before:'] = tokenized_sentence[index-1]
            if len(tokenized_sentence)==3: 
                features['one token before:'] = tokenized_sentence[index-1]
                features['one token after:'] = tokenized_sentence[index+1]
            if len(tokenized_sentence)>3: 
                features['one token before:'] = tokenized_sentence[index-1] 
                features['one token after:'] = tokenized_sentence[index+1]
                features['two tokens after:'] = tokenized_sentence[index+2] 
        if index>1: 
            if index==2:
                if len(tokenized_sentence) == 3:
                    features['one token before:'] = tokenized_sentence[index-1]
                    features['two tokens before:'] = tokenized_sentence[index-2]
                    features['is 3rd token'] = (index==2)
                if len(tokenized_sentence)==4: 
                    features['one token before:'] = tokenized_sentence[index-1]
                    features['two tokens before:'] = tokenized_sentence[index-2], 
                    features['one token after:'] = tokenized_sentence[index+1]
            if index<(len(tokenized_sentence)-2): 
                features['one token before:'] = tokenized_sentence[index-1]
                features['two tokens before:'] = tokenized_sentence[index-2]
                features['one token after:'] = tokenized_sentence[index+1]
                features['two tokens after:'] = tokenized_sentence[index+2] 
            if index==(len(tokenized_sentence)-2): 
                features['one token before:'] = tokenized_sentence[index-1]
                features['two tokens before:'] = tokenized_sentence[index-2]
                features['one token after:'] = tokenized_sentence[index+1]
                features['is penultimate token'] = (index == len(tokenized_sentence)-2)
            if index==(len(tokenized_sentence)-1):
                features['one token before:'] = tokenized_sentence[index-1]
                features['two tokens before:'] = tokenized_sentence[index-2]
        features['is last token'] = (index == len(tokenized_sentence)-1)     

        return features


    def predict_token(self, tokenized_sentence, index, unidec_melliza):
        
        self.tokenized_sentence = tokenized_sentence 
        self.index = index
        self.unidec_melliza = unidec_melliza
        
        sentence_features = Diacriticize.extract_features(self.tokenized_sentence, self.index)
        predicted_token = self.melliza_and_clsfr_dict[self.unidec_melliza].classify(sentence_features)
        
        return predicted_token

    
    def predict_sentence(self, tokenized_sentence): 
        
        self.tokenized_sentence = tokenized_sentence 

        predicted_toks = []
        for (index,token) in enumerate(self.tokenized_sentence): 
            unidec_melliza = unidecode.unidecode(token.casefold())
            if unidec_melliza in self.melliza_and_clsfr_dict:
                predicted_toks.append(self.predict_token(self.tokenized_sentence, index, unidec_melliza))
            elif unidec_melliza in self.invariantly_diacriticized_tokens_dict: 
                predicted_toks.append(self.invariantly_diacriticized_tokens_dict[unidec_melliza])
            elif 'ion' in unidec_melliza[-3:]:
                predicted_toks.append(unidec_melliza.replace('ion', 'ión'))
            elif 'patia' in unidec_melliza[-5:]:
                predicted_toks.append(unidec_melliza.replace('patia', 'patía'))
            elif 'patias' in unidec_melliza[-6:]:
                predicted_toks.append(unidec_melliza.replace('patias', 'patías'))  
            elif 'grafía' in unidec_melliza[-6:]: 
                predicted_toks.append(unidec_melliza.replace('grafia', 'grafía'))
            elif 'grafías' in unidec_melliza[-7:]: 
                predicted_toks.append(unidec_melliza.replace('grafias', 'grafías'))
            elif 'latria' in unidec_melliza[-5:]:
                predicted_toks.append(unidec_melliza.replace('latria', 'latría'))
            elif 'eria' in unidec_melliza[-4:]:
                predicted_toks.append(unidec_melliza.replace('eria', 'ería'))
            elif 'erias' in unidec_melliza[-5:]:
                predicted_toks.append(unidec_melliza.replace('erias', 'erías'))
            elif 'latrias' in unidec_melliza[-6:]:
                predicted_toks.append(unidec_melliza.replace('latrias', 'latrías'))
            elif 'tomia' in unidec_melliza[-5:]:
                predicted_toks.append(unidec_melliza.replace('tomia', 'tomía'))
            elif 'tomias' in unidec_melliza[-6:]:
                predicted_toks.append(unidec_melliza.replace('tomias', 'tomías'))
            elif 'logia' in unidec_melliza[-5:]:
                predicted_toks.append(unidec_melliza.replace('logia', 'logía'))
            elif 'logias' in unidec_melliza[-6:]:
                predicted_toks.append(unidec_melliza.replace('logias', 'logías'))
            elif 'abamos' in unidec_melliza[-6:]:
                predicted_toks.append(unidec_melliza.replace('abamos', 'ábamos'))
            elif 'america' in unidec_melliza[-7:]:
                predicted_toks.append(unidec_melliza.replace('america', 'américa'))
            elif 'americas' in unidec_melliza[-8:]:
                predicted_toks.append(unidec_melliza.replace('americas', 'américas'))    
            elif 'logico' in unidec_melliza[-6:]:
                predicted_toks.append(unidec_melliza.replace('logico', 'lógico'))
            elif 'fobico' in unidec_melliza[-6:]:
                predicted_toks.append(unidec_melliza.replace('fobico', 'fóbico'))
            elif 'fobicos' in unidec_melliza[-6:]:
                predicted_toks.append(unidec_melliza.replace('fobicos', 'fóbicos'))
            elif 'logicos' in unidec_melliza[-8:]:
                predicted_toks.append(unidec_melliza.replace('logicos', 'lógicos'))
            elif 'zon' in unidec_melliza[-3:]:
                predicted_toks.append(unidec_melliza.replace('zon', 'zón'))
            elif 'zones' in unidec_melliza[-5:]:
                predicted_toks.append(unidec_melliza.replace('zones', 'zónes'))
            elif 'scopico' in unidec_melliza[-7:]:
                predicted_toks.append(unidec_melliza.replace('scopico', 'scópico'))
            elif 'scopicos' in unidec_melliza[-7:]:
                predicted_toks.append(unidec_melliza.replace('scopicos', 'scópicos'))
            elif 'onimo' in unidec_melliza[-5:]:
                predicted_toks.append(unidec_melliza.replace('onimo', 'ónimo'))
            elif 'onimos' in unidec_melliza[-6:]:
                predicted_toks.append(unidec_melliza.replace('onimos', 'ónimos'))
            elif 'onicos' in unidec_melliza[-6:]:
                predicted_toks.append(unidec_melliza.replace('onicos', 'ónicos'))
            elif 'onicas' in unidec_melliza[-6:]:
                predicted_toks.append(unidec_melliza.replace('onicas', 'ónicas'))
            elif 'onica' in unidec_melliza[-5:]:
                predicted_toks.append(unidec_melliza.replace('onica', 'ónica'))
            elif 'onico' in unidec_melliza[-5:]:
                predicted_toks.append(unidec_melliza.replace('onico', 'ónico'))
            elif 'ectomias' in unidec_melliza[-8:]: 
                predicted_toks.append(unidec_melliza.replace('ectomias', 'ectomías'))
            elif 'ectomia' in unidec_melliza[-7:]: 
                predicted_toks.append(unidec_melliza.replace('ectomia', 'ectomía'))
            elif 'cigotico' in unidec_melliza[-8:]: 
                predicted_toks.append(unidec_melliza.replace('cigotico', 'cigótigo'))
            elif 'cigoticos' in unidec_melliza[-9:]: 
                predicted_toks.append(unidec_melliza.replace('cigoticos', 'cigótigos'))     
            elif 'centrico' in unidec_melliza[-9:]: 
                predicted_toks.append(unidec_melliza.replace('centrico', 'céntrico'))
            elif 'centricos' in unidec_melliza[-9:]: 
                predicted_toks.append(unidec_melliza.replace('centricos', 'céntricos'))  
            elif 'aceo' in unidec_melliza[-4:]: 
                predicted_toks.append(unidec_melliza.replace('aceo', 'áceo'))
            elif 'aceos' in unidec_melliza[-4:]: 
                predicted_toks.append(unidec_melliza.replace('aceos', 'áceos'))
            elif 'orico' in unidec_melliza[-5:]: 
                predicted_toks.append(unidec_melliza.replace('orico', 'órico'))
            elif 'oricos' in unidec_melliza[-5:]: 
                predicted_toks.append(unidec_melliza.replace('oricos', 'óricos'))
            elif 'iendose' in unidec_melliza[-7:]: 
                predicted_toks.append(unidec_melliza.replace('iendose', 'iéndose'))
            elif 'ificamente' in unidec_melliza[-7:]: 
                predicted_toks.append(unidec_melliza.replace('ificamente', 'íficamente'))           
            else: 
                predicted_toks.append(unidec_melliza)
            
        return predicted_toks

if __name__ =='__main__': 
    parser = argparse.ArgumentParser(description='Diacriticizes a sequence of unidecoded Spanish tokens.')
    parser.add_argument('tokens', help='a string sequence of unidecoded Spanish tokens without quotes')
    args = parser.parse_args()

    #predicted_tokens = Diacriticize.predict_sentence(nltk.word_tokenize(str(args.tokens)))
    #rejoined_tokens = ' '.join(predicted_tokens)
    #print(rejoined_tokens)

    #predicted_tokens = Diacriticize.predict_sentence(nltk.word_tokenize(str(args.tokens)))
    #glue_punct = ''.join(predicted_tokens[-2:])
    #predicted_tokens[-2] = glue_punct
    #predicted_tokens.pop()
    #print(' '.join(predicted_tokens))

    predicted_tokens = Diacriticize().predict_sentence(nltk.word_tokenize(str(args.tokens)))
    glue_punct = ''.join(predicted_tokens[-2:])
    predicted_tokens[-2] = glue_punct
    predicted_tokens.pop()
    i=0
    for token, predicted_token in zip(nltk.word_tokenize(str(args.tokens)), predicted_tokens): 
        if token == token.capitalize(): 
            predicted_tokens[i] = predicted_token.capitalize()
        else: 
            continue 
        i+=1
    print(' '.join(predicted_tokens))

