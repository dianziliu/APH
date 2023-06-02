import pickle
import os
import sys
import pandas    as pd
import nltk
from nltk.corpus import wordnet as wn
from collections import defaultdict


paths = [
        'Musical_Instruments_5.json',
        "Video_Games_5.json",
        "Sports_and_Outdoors_5.json",
        "Office_Products_5.json",
        'Grocery_and_Gourmet_Food_5.json'    
    ]
data_path_fmt="amazon_data/{}"
save_path_fmt="ASPE/aspect_res/res2/{}"


syn2unq=dict()



def read_csv2dict(path):
    df=pd.read_csv(path)
    d={}
    for idx,row in df.iterrows():
        word,frq=row['dep'],row['frq']
        d[word]=frq
    return d
def get_original_word(word,d):
    # 得到合并后的词
    if word in syn2unq:
        unq= syn2unq[word]
    else:
        conditions=wn.synsets(word)
        syns=[a for i in conditions for a in i.lemma_names()]
        syns.append(word)
        syns_frq=[]
        for a in syns:
            if a in d:
                syns_frq.append([a,d[a]])
        syns_frq=sorted(syns_frq,key=lambda x: x[1],reverse=True)
        try:
            unq=syns_frq[0][0]
        except:
            unq=word
        syn2unq[word]=unq
    return unq




def merge(old_path,new_path):
    with open(old_path,"rb") as f:
        old_res=pickle.load(f)
    dir=os.path.dirname(old_path)
    d=read_csv2dict(os.path.join(dir,"a_count.csv"))
    new_res=[]
    a_count=defaultdict(int)
    for one in old_res:
        tmp=[]
        for a,s in one:
            a_=get_original_word(a,d)
            a_count[a_]+=1
            tmp.append((a,a_,s))
        new_res.append(tmp)
    
    with open(os.path.join(dir,"a_count2.csv"),"wb") as f:
        pickle.dump(a_count,f)    
    with open(new_path,"wb") as f:
        pickle.dump(new_res,f)

def proceess_all_datasets():
    ###################  1. 数据集处理  ###################
    paths = [
        # "Automotive_5.json",
        # "Digital_Music_5.json",
        # 'Musical_Instruments_5.json',
        # "Pet_Supplies_5.json",
        # "Sports_and_Outdoors_5.json",
        "Toys_and_Games_5.json",
        # "Tools_and_Home_Improvement_5.json",
        # "Office_Products_5.json",
        # 'Grocery_and_Gourmet_Food_5.json',
        # "Video_Games_5.json", 
        # "Beauty_5.json",
        # "yelp2.json"

    ]
    save_dir="data/"
    data_path_fmt=save_dir+"{}/as.pkl"
    save_path_fmt=save_dir+"{}/as2.pkl"

    ###################  2. 处理逻辑  ###################
    for p in paths:
        p=p[:-5]
        data_path=data_path_fmt.format(p)
        save_path=save_path_fmt.format(p)
        # count_deps(data_path,save_path)
        ###################  处理逻辑  ###################
        # count_deps_mt(data_path,save_path)
        merge(data_path,save_path)


if __name__=="__main__":

    proceess_all_datasets()

