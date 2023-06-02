import os
import pickle
import pandas as pd

import nltk
from nltk.corpus import wordnet as wn



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


def loadText2dict(path,sep=",",head=True):
    d={}
    # with open(path,"r") as f:
    #     if head:
    #         f.readline()
    #     for line in f.readlines():
    #         a,n=line.strip().split(sep)
    #         d[a]=int(n)
    df=pd.read_csv(path)
    for dix,row in df.iterrows():
        a,n=row['dep'],row['frq']
    return d

def filter(c1,c2,dir,new_path):
    pkl_path=dir+"as2.pkl"
    with open(pkl_path,"rb") as f:
        old_res=pickle.load(f)

    with open(os.path.join(dir,"a_count2.csv"),"rb") as f:
        a_df=pickle.load(f)
    new_res=[]
    # a_df=loadText2dict(dir+"a_count.csv")
    # s_df=loadText2dict(dir+"s_count.csv")
    for one in old_res:
        tmp=[]
        for a,a_,s in one:
            if a_df[a_]>c1:
                tmp.append((a,a_,s))
            else:
                continue
        new_res.append(tmp)

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
    data_path_fmt=save_dir+"{}/"
    save_path_fmt=save_dir+"{}/as3.pkl"

    ###################  2. 处理逻辑  ###################
    for p in paths:
        p=p[:-5]
        data_path=data_path_fmt.format(p)
        save_path=save_path_fmt.format(p)
        # count_deps(data_path,save_path)
        ###################  处理逻辑  ###################
        # count_deps_mt(data_path,save_path)
        filter(10,10,data_path,save_path)

if __name__=="__main__":
    
    proceess_all_datasets()
