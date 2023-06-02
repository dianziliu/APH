from collections import defaultdict
import pickle
import os

class Opinion_lexion_classifer:
    """
    一个包装类
    
    """
    support_pos=["noun","adj","adv","verb"]

    def __init__(self,dir) -> None:
        d_path=dir+"/opinion_lexion_classifer.pkl"
        if os.path.exists(d_path):
            with open(d_path,"rb") as f:
                d=pickle.load(f)
        else:
            d = self.load_lexion_from_file(dir)  
            with open(d_path,"wb") as f:
                pickle.dump(d,f)
        self.d=d

    def load_lexion_from_file(self, dir):
        d=defaultdict(dict)
        file_fmt="{}/{}/{}.txt"
        sents=[(1,"postive"),(-1,"negtive")]
        for score, sen in sents:
            for pos in ["noun","adj","adv","verb"]:
                path=file_fmt.format(dir,sen,pos)
                with open(path,"r") as f:
                    for line in f.readlines():
                        word=line.strip().lower()
                        d[word][pos]=score
        return d

    def sentiment_classifer(self,token,Pos):
        res=self.d[token.lower()]
        if Pos in res:
            return res[Pos]
        elif "adj" in res:
            return res["adj"]
        elif "adv" in res:
            return res["adv"]
        elif "verb" in res:
            return res["verb"]
        return 0
    def predict(self,tokens):
        res=[self.sentiment_classifer(token,"noun ") for token in tokens]
        return res
if __name__=="__main__":
    a=Opinion_lexion_classifer("OpinionLexion")
    print(
        a.sentiment_classifer("elimination","noun")
    )
    