import itertools
import os
import pickle
import re
import sys
from collections import Counter
from this import d
from tokenize import group

import networkx as nx
import nltk
import numpy as np
import pandas as pd
import scipy.sparse as sp
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.data.path.append("nltk_data")
# data_type = sys.argv[1]
ps = PorterStemmer()
# tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
stopWords = set(stopwords.words('english'))
tags = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS',
        'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'PRP']


def ensureDir(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)


class data_process(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def data_review(self, ui_reviews):
        def clean_str(string):
            string = re.sub(r"[^A-Za-z]", " ", string)
            tokens = [w for w in word_tokenize(string.lower())]
            return tokens
        ui_text = {}
        for ui, reviews in ui_reviews.items():
            for review in reviews:
                s = clean_str(review)
                if int(ui) in ui_text:
                    ui_text[int(ui)].append(s)
                else:
                    ui_text[int(ui)] = [s]
        return ui_text


    def data_load(self, data):
        uid, iid, rate = [], [], []
        for line in data:
            line = line.split(',')
            uid.append(int(line[0]))
            iid.append(int(line[1]))
            rate.append(float(line[2]))
        return uid, iid, rate
    def load2id(self):
        dir=os.path.join(self.data_dir,"pro_data")
        with open(os.path.join(dir,"user2id"),"rb")  as f:
            self.user2id=pickle.load(f)
        with open(os.path.join(dir,"item2id"),"rb")  as f:
            self.item2id=pickle.load(f)   


    def generate_graph(self,file):
        def encode_word(a):
            # 对a Jinx编码
            if a not in self.word2id:
                self.word2id[a] = self.w_cnt
                self.w_cnt += 1
            word1_index = self.word2id[a]    
            return word1_index  
        def g_graph(df,group_key):
            num_node = 300
            reviews_graphs = {}
            for user,group in df.groupby(group_key):
                G=nx.DiGraph()
                c=Counter()
                edge_list=set()
                for idx,row in group.iterrows():
                    u,i,a,s=row.values
                    u=self.user2id[u]
                    i=self.item2id[i]
                    word1_index=encode_word(a)
                    c.update([word1_index])
                    if group_key=="user":
                        side=u
                    else:
                        side=i
                    edge_list.add((side,word1_index,int(s)+1))
                G.add_weighted_edges_from(list(edge_list))

                node_list = np.array([x[0]
                                      for x in c.most_common(num_node)])
                # 稀疏表示图
                adj_matrix = nx.to_scipy_sparse_matrix(G, nodelist=node_list)
                # adj_matrix = nx.to_scipy_sparse_array(G, nodelist=node_list)

                index_val = sp.find(adj_matrix)
                edg_index = np.array([index_val[1], index_val[0]])
                edg_value = np.array(index_val[2])
                if group_key=="user":
                    idx=self.user2id[user]
                else:
                    idx=self.item2id[user]
                reviews_graphs[idx] = (edg_index, edg_value, node_list)    
            return reviews_graphs
        ####################   1. 准备数据     ####################
        aspect_df=pd.read_csv(file,names=["user","item","aspect","sentiment"])
        self.word2id = dict()
        self.w_cnt = 0
        ####################   2. 构建用户图和物品图     ####################
        u_graphs = g_graph(aspect_df,"user")
        i_graphs = g_graph(aspect_df,"item")
        assert len(self.word2id) == self.w_cnt
        return u_graphs, i_graphs        

    def process_d(self,kg_file):
        prodata = 'pro_data'
        train_data, test_data, valid_data, user_reviews, item_reviews = self.open_files(prodata)
        self.load2id()

        # shuffle data and select train set,test set and validation set

        print('load rating data...')
        uid_train, iid_train, rate_train = self.data_load(train_data)
        uid_valid, iid_valid, rate_valid = self.data_load(valid_data)
        uid_test, iid_test, rate_test = self.data_load(test_data)
        num_rating = len(rate_train) + len(rate_test) + len(rate_valid)


        print('splitting reviews...')
        self.u_text = self.data_review(user_reviews)
        self.i_text = self.data_review(item_reviews)
        self.user_num = len(self.u_text)
        self.item_num = len(self.i_text)

        print('generating graph of reviews')
        u_graphs, i_graphs = self.generate_graph(kg_file)

        print('number of users:', self.user_num)
        print('number of items:', self.item_num)
        print('number of ratings:', num_rating)
        print('number of words', len(self.word2id))
        para = {}
        para['user_num'] = self.user_num
        para['item_num'] = self.item_num
        para['rate_num'] = num_rating
        para['vocab'] = self.word2id
        para['train_length'] = len(rate_train)
        para['eval_length'] = len(rate_valid)
        para['test_length'] = len(rate_test)
        print('write begin')
        d_train = list(zip(uid_train, iid_train, rate_train))
        d_valid = list(zip(uid_valid, iid_valid, rate_valid))
        d_test = list(zip(uid_test, iid_test, rate_test))
        train_path = open(os.path.join(
            self.data_dir, 'data.train'), 'wb')
        pickle.dump(d_train, train_path)
        valid_path = open(os.path.join(
            self.data_dir, 'data.eval'), 'wb')
        pickle.dump(d_valid, valid_path)
        test_path = open(os.path.join(
            self.data_dir, 'data.test'), 'wb')
        pickle.dump(d_test, test_path)
        para_path = open(os.path.join(
            self.data_dir, 'data.para'), 'wb')
        pickle.dump(para, para_path)
        u_graph_path = open(os.path.join(
            self.data_dir, 'data.user_graphs'), 'wb')
        pickle.dump(u_graphs, u_graph_path)
        i_graph_path = open(os.path.join(
            self.data_dir, 'data.item_graphs'), 'wb')
        pickle.dump(i_graphs, i_graph_path)
        print('done!')

    def open_files(self, prodata):
        train_data = open(os.path.join(
            self.data_dir + prodata + '/data_train.csv'), 'r')
        test_data = open(os.path.join(
            self.data_dir + prodata + '/data_test.csv'), 'r')
        valid_data = open(os.path.join(
            self.data_dir + prodata + '/data_valid.csv'), 'r')
        user_reviews = pickle.load(
            open(os.path.join(self.data_dir + prodata + '/user_review'), 'rb'))
        item_reviews = pickle.load(
            open(os.path.join(self.data_dir + prodata + '/item_review'), 'rb'))
            
        return train_data,test_data,valid_data,user_reviews,item_reviews



def main_for_sig():
    # 怎么做数据隔离呢？
    pass

if __name__ == '__main__':
    # np.random.seed(2019)
    # random.seed(2019)
    # Data_process = data_process('../data/' + data_type + '/')
    paths = [
        # 'Musical_Instruments_5',
        # "Office_Products_5",
        # "Toys_and_Games_5",
        # "Video_Games_5",
        # "Automotive_5",
        # "Digital_Music_5",
        # "Pet_Supplies_5",
        # "Sports_and_Outdoors_5",
        # "Tools_and_Home_Improvement_5",
        # "Beauty_5",
        "yelp2"
    ]

    for p in paths:
        Data_process=data_process("data/{}/".format(p))
        kg_path="data/{}/as_kg.txt".format(p)
        Data_process.process_d(kg_path)
