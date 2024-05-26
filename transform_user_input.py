class MyClass:
    def __init__(self, dictionary):
        self.dictionary = dictionary
        self.create_data = self.create_formatted_data(dictionary)
        self.features = self.build_features(self.create_data)
        self.labels = self.build_labels(self.create_data) # y labels
        self.vocab = set()
        self.idf = {}
        self.user_vocab = set()
        self.user_idf = {}
        self.fit(self.features)
        self.x_features = self.transform(self.features) # X features 
        self.user_input(self.features)

    def create_formatted_data(self,dictionary):
        training_dict = {}
        for intent, question_list in dictionary.items():
            
            for question in question_list:
                training_dict[question] = intent
                
        return training_dict
    
    def build_features(self,create_data):
        feature = np.array(list(create_data.keys()))
        return feature
    
    def build_labels(self,create_data):
        labels = np.array(list(create_data.values()))
        reshaped_labels = labels.reshape(-1,1)
        return reshaped_labels
    
    def fit(self, documents):
        self.doc_count = len(documents)
        # Build vocabulary and compute document frequency (DF)
        for doc in documents:
            terms = set(doc.split())
            self.vocab.update(terms)
            
            for term in terms:
                if term in self.idf:
                    self.idf[term] += 1
                else:
                    self.idf[term] = 1
        
        # Compute IDF
        for term in self.idf:
            self.idf[term] = math.floor(math.log(self.doc_count / (1 + self.idf[term])))
        
    def transform(self, documents):
        tfidf_matrix = np.zeros((len(documents),len(self.vocab))) 
        term_index = {term: i for i, term in enumerate(sorted(self.vocab))}

        for i, doc in enumerate(documents):
            term_freq = Counter(doc.split())
            doc_len = len(doc.split())
            
            for term in term_freq:
                if term in self.vocab:
                    tf = term_freq[term] / doc_len
                    tfidf_matrix[i, term_index[term]] = tf * self.idf[term]

        return tfidf_matrix    

    def user_input(self,documents):
        while True:
            self.user_data = input("User>> ")
            self.transform_user_input(documents)
            if self.user_data == "quit":
                break

    def transform_user_input(self,documents):
        tfidf_matrix = np.zeros((1, len(documents)))
        term_index = {term: i for i, term in enumerate(sorted(self.vocab))}
        
        if self.user_data in self.vocab:
            tfidf_matrix[0, term_index[self.user_data]] = 1
        else:
            tfidf_matrix[0, 5] = 0
        
        return tfidf_matrix

import json
import numpy as np
import math
from collections import Counter

with open("chatdata.json", 'r') as f:
    chat_data = json.load(f)
    f.close()

MyClass(chat_data)


