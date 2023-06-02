
import json
import os
import pickle
from cgitb import text
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import nltk
import pandas as pd
import spacy
from spacy import displacy
from tqdm import tqdm


nlp = spacy.load("en_core_web_sm")
from nltk.stem.wordnet import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
amod_count=0
nsubj_count=0
aspect_count=defaultdict(int)
sentiment_count=defaultdict(int)


STOPWORDS=[
    "one","daughter","son","kid","child", " ","boy","grandson","granddaughter","guy"
]

def sort_dict(d):
    l=[(k,v) for k,v in d.items()]
    sorted_l=sorted(l,reverse=True,key=lambda x: x[1])
    return sorted_l


def store_list(l,save_path):
    d = {"dep": [i[0] for i in l],
         "frq": [i[1] for i in l]}

    df=pd.DataFrame(d)
    df.to_csv(save_path,index=False)

def readfile(filename):
	f = open(filename, encoding='utf-8')
	data = []
	for line in f.readlines():
		line = json.loads(line)
		data.append(line['reviewText'])
	f.close()
	return data

def lemmatize(word):
    ###################  2. 词性归并  ###################  
    # text=lemmatizer.lemmatize(token.lemma_,pos="n")
    # head=lemmatizer.lemmatize(token.head.lemma_,pos="n")
    return  lemmatizer.lemmatize(word,pos="n")


def dep_check(st):
    """
    需要明确dep_rule的规则
    aspect词的位置
    sentiment词的位置
    """
    aspects_sentiments=[]
    # 
    # for one_word in st:
    #     if one_word["dep"]=="amod":
    #         aspects_sentiments.append(
    #             one_word["head"],one_word["text"]
    #         )
    global nsubj_count,amod_count,aspect_count
    for idx,one_word in enumerate(st):
        if one_word["dep"]=="nsubj":
            b=one_word["head"]
            for c in st[idx:]:
                if c["dep"]=="acomp" and c["head"]==b:  
                    aspects_sentiments.append(
                        (lemmatize(one_word["text"]),lemmatize(c["text"]))
                    )
                    nsubj_count+=1
                    break
                    
    for idx,one_word in enumerate(st):
        
        if one_word["dep"]=="amod":
            # 组合规则
            c=one_word["head"]
            for c_ in st[idx:]:
                if c_["dep"]=="dobj" and c_["text"]==c:
                    aspects_sentiments.append(
                        ("{}_{}".format(lemmatize(c_["head"]),lemmatize(c))
                        ,lemmatize(one_word["text"]))
                    )
                    break
            # 独立规则
            else:
                 aspects_sentiments.append(
                (lemmatize(one_word["head"]),lemmatize(one_word["text"]))
            )
            amod_count+=1
    final_aspects_sentiments=[]
    # 手动过滤掉一些停用词
    for a,s in aspects_sentiments:
        if a in STOPWORDS:
            continue
        aspect_count[a]+=1
        sentiment_count[s]+=1
        final_aspects_sentiments.append((a,s))
    return final_aspects_sentiments


def dep_check3(st,dep_rules,mode,asepct_site,sentiment_site):
    """
    mode: 0 表示链式关系，1 便是其他关系
    """
    aspects_sentiments=[]
    for idx,one_word in enumerate(st):
        if one_word["dep"]==dep_rules[0]:
            a=one_word["text"]
            b=one_word["head"]
            if mode==0:
                
                for b_ in st[idx:]:
                    if b_["dep"]==dep_rules[1] and b_["text"]==b:
                        c=b_["head"]
                        
                        break
            # 独立规则
            else:
                for c_ in st[idx:]:
                    if c_["dep"]==dep_rules[1] and c_["head"]==b:
                        c=c_["text"]
                        break    
            res=[a,b,c]
            aspects_sentiments.append(
                "_".join(res[asepct_site],res[sentiment_site])
                ,one_word["text"]
                )


def get_aspect_sentiment(doc):
    """
        依存关系分析
        多进程中的计算单元
        在这里定义需要保留的元素
    """
    res=nlp(doc)
    txt=[]
    for token in res:

        ###################  1. 过滤  ###################
        if token.is_stop or not token.is_alpha:
            continue

        ###################  2. 数据打包  ###################  
        txt.append(
            {"text":token.lemma_,
            "dep":token.dep_,
            "head":token.head.lemma_,
            }
        )
    return dep_check(txt)


def count_deps_mt(path,save_path):
    """
        多进程的模型，可以运行的更快一些
    """
    print("now start to process ", path)
    data=readfile(path)

    ###################  1. 多线程处理  ###################
    with ProcessPoolExecutor(10) as exectuor:
        res=list(
            tqdm(exectuor.map(get_aspect_sentiment,data,chunksize=128),ncols=100,total=len(data))
        )

    ###################  2. 结果分析和保存  ###################
    # 每一行针对一条评论，[(aspect_word, sentiment_word),...]
    save_aspect_sentiments(res,save_path)

def count_deps(path):
    """
        多进程的模型，可以运行的更快一些
    """
    print("now start to process ", path)
    data=readfile(path)
    res=[]
    for doc in data:
        get_aspect_sentiment(doc)

    ###################  1. 多线程处理  ###################
    # with ProcessPoolExecutor(10) as exectuor:
    #     res=list(
    #         tqdm(exectuor.map(get_aspect_sentiment,data,chunksize=128),ncols=100,total=len(data))
    #     )

    ###################  2. 结果分析和保存  ###################
    # 每一行针对一条评论，[(aspect_word, sentiment_word),...]
    # save_aspect_sentiments(res,save_path)


def save_aspect_sentiments(res,save_path):
    a_count=defaultdict(int)
    s_count=defaultdict(int)
    for one in res:
        for a,s in one:
            a_count[a]+=1
            s_count[s]+=1
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    s1=save_path+"/as.pkl"
    s2=save_path+"/a_count.csv"
    s3=save_path+"/s_count.csv"
    with open(s1,"wb") as f:
        pickle.dump(res,f)
    store_list(sort_dict(a_count),s2)
    store_list(sort_dict(s_count),s3)

def count_deps(path,save_path):
    print("now start to process ", path)
    data=readfile(path)
    DC_dict=defaultdict(int)
    for doc in tqdm(data,ncols=100):
        get_aspect_sentiment(doc)
    print(amod_count)
    print(nsubj_count)
    print(aspect_count)
    print(sentiment_count)

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
        # "Beauty_5.json"
        # "yelp2.json"
    ]
    data_path_fmt="data/{}"
    save_path_fmt="data/{}"

    ###################  2. 处理逻辑  ###################
    for p in paths:
        data_path=data_path_fmt.format(p)
        save_path=save_path_fmt.format(p[:-5])
        # count_deps(data_path,save_path)
        ###################  处理逻辑  ###################
        count_deps_mt(data_path,save_path)

if __name__=="__main__":

    proceess_all_datasets()
    # count_deps("amazon_data/Toys_and_Games_5.json","")
