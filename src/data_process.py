import os
import numpy as np
import json
import pandas as pd
import nltk
from nltk.corpus import stopwords
import pickle
import random
import sys
from nltk.stem import PorterStemmer

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
        self.u_text = dict()
        self.i_text = dict()

    def numb_id(self, data):
        uid = []
        iid = []
        for x in data['users_id']:
            uid.append(self.user2id[x])
        for x in data['items_id']:
            iid.append(self.item2id[x])
        data['users_id'] = uid
        data['items_id'] = iid
        return data

    def data_review(self, train_data):
        # 构建历史记录和评论的字典
        user_rid = {}
        item_rid = {}
        user_reviews = {}
        item_reviews = {}
        # u,i,reiew,rating
        for line in train_data.values:
            # uid
            if int(line[0]) in user_reviews:
                user_reviews[int(line[0])].append(line[2])
                user_rid[int(line[0])].append(int(line[1]))
            else:
                user_reviews[int(line[0])] = [line[2]]
                user_rid[int(line[0])] = [int(line[1])]
            if int(line[1]) in item_reviews:
                item_reviews[int(line[1])].append(line[2])
                item_rid[int(line[1])].append(int(line[0]))
            else:
                item_reviews[int(line[1])] = [line[2]]
                item_rid[int(line[1])] = [int(line[0])]
        assert len(user_reviews) == len(self.user2id)
        assert len(item_reviews) == len(self.item2id)
        return user_reviews, item_reviews, user_rid, item_rid

    def data_load(self, data):
        uid = data['users_id'].values
        iid = data['items_id'].values
        rate = data['rates'].values
        return uid, iid, rate

    def process_d(self):
        def get_count(data, id):
            data_groupby = data.groupby(id, as_index=False)
            return data_groupby.size()


        ####################  1. 读取json文件，并转化为df格式   ['users_id', 'items_id', 'reviews', 'rates']   ####################
        Data_file = os.path.join(os.path.dirname(os.path.dirname(self.data_dir)), 'data.json')
        f = open(Data_file)
        data = self.get_data_df(f)

        data = self.transID(get_count, data)

        # 8/2/1
        train_data, tv_data, valid_data = self.split_data(data)

        # get reviews of each user and item
        print('start collect reviews for users and items...')
        user_reviews, item_reviews, user_rid, item_rid = self.data_review(
            train_data)

        
        print('start saving...')
        train_data1 = train_data[['users_id', 'items_id', 'rates']]
        test_data2 = tv_data[['users_id', 'items_id', 'rates']]
        valid_data1 = valid_data[['users_id', 'items_id', 'rates']]
        train_data1.to_csv(os.path.join(
            self.data_dir, 'data_train.csv'), index=False, header=None)
        test_data2.to_csv(os.path.join(
            self.data_dir, 'data_test.csv'), index=False, header=None)
        valid_data1.to_csv(os.path.join(
            self.data_dir, 'data_valid.csv'), index=False, header=None)
        pickle.dump(user_reviews, open(
            os.path.join(self.data_dir, 'user_review'), 'wb'))
        pickle.dump(item_reviews, open(
            os.path.join(self.data_dir, 'item_review'), 'wb'))
        pickle.dump(user_rid, open(os.path.join(
            self.data_dir, 'user_rid'), 'wb'))
        pickle.dump(item_rid, open(os.path.join(
            self.data_dir, 'item_rid'), 'wb'))
        # 存储
        pickle.dump(self.user2id, open(os.path.join(
            self.data_dir, 'user2id'), 'wb'))
        pickle.dump(self.item2id, open(os.path.join(
            self.data_dir, 'item2id'), 'wb'))
        print('done!')

    def split_data(self, data):
        """
        这块算起来太慢了
        """
        train_df = pd.DataFrame(
            columns=['users_id', 'items_id', 'reviews', 'rates'])

        # 确保每个用户和物品出现1次
        for user in range(len(self.user2id)):
            if user not in train_df['users_id'].values:
                ddf = data[data.users_id.isin([user])].iloc[[0]]
                # train_df = train_df.append(ddf)
                train_df=pd.concat([train_df,ddf])
                data.drop(ddf.index, inplace=True)
                
        for item in range(len(self.item2id)):
            if item not in train_df['items_id'].values:
                ddf = data[data.items_id.isin([item])].iloc[[0]]
                # train_df = train_df.append(ddf)
                train_df=pd.concat([train_df,ddf])
                data.drop(ddf.index, inplace=True)

        print('start splitting dataset...')
        # shuffle data and select train set,test set and validation set
        data_len = data.shape[0]
        index = np.random.permutation(data_len)
        data = data.iloc[index]
        train_data = data.head(int(data_len * 0.8) - train_df.shape[0])
        train_data = pd.concat([train_data, train_df], axis=0)
        # 这里处理问题。
        tv_data = data.tail(int(data_len * 0.2))
        valid_data = tv_data.head(int(data_len * 0.1))

        return train_data,tv_data,valid_data

    def transID(self, get_count, data):
        """ 将用户和物品ID进行编码"""
        # 感觉这块卡计算
        # users_count = get_count(data, 'users_id')
        # items_count = get_count(data, 'items_id')
        # unique_users = users_count.index
        # unique_items = items_count.index
        unique_users = data.users_id.unique().tolist()
        unique_items = data.items_id.unique().tolist()
        self.user2id = dict((x, i) for (i, x) in enumerate(unique_users))
        self.item2id = dict((x, i) for (i, x) in enumerate(unique_items))

        data = self.numb_id(data)
        return data
    

    def get_data_df(self, f):
        """
        ['users_id', 'items_id', 'reviews', 'rates']
        """
        users_id = []
        items_id = []
        reviews = []
        rates = []
        print('start extracting data...')
        for line in f:
            js = json.loads(line)
            if str(js['reviewerID']) == 'unknow':
                continue
            if str(js['asin']) == 'unknow':
                continue
            users_id.append(str(js['reviewerID']))
            items_id.append(str(js['asin']))
            reviews.append(js['reviewText'])
            rates.append(js['overall'])
        
        data = pd.DataFrame({'users_id': users_id, 'items_id': items_id, 'reviews': reviews, 'rates': rates})[
            ['users_id', 'items_id', 'reviews', 'rates']]
        print('number of interaction:', data.shape[0])
        return data


if __name__ == '__main__':
    np.random.seed(2020)
    random.seed(2020)
    # path = '../data/' + data_type + '/pro_data/'
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
        path="data/{}/pro_data/".format(p) 
        ensureDir(path) # 创建一个文件夹
        Data_process = data_process(path)
        Data_process.process_d()
