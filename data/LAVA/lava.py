import torch 
import json 
import os 
import random 
from transformers import CLIPProcessor
from PIL import Image 
from itertools import combinations 

class LAVA(torch.utils.data.Dataset):
    def __init__(self, k):
        super().__init__()
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.path = os.path.join(cur_dir, 'lava.json')
        self.listdir = os.path.join(cur_dir, 'images')
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32") 
        self.lava = json.load(open(self.path, 'r'))
        self.res = dict()  
        
        
        """
        As the bloody dataset lacks some filenames, we need to add them manually.
        Death to Danny, Andrei, and Evgeni  
        """
        tempdict = {} 
        for each in self.lava:
            id = str(each['sentenceID']) + ',' + str(each['variantID'])
            tempdict[id] = each
        

        """
        Buildingup Minibatches.... 
        
        Buildingup Principles
        
        1. Contrastvariants stick together
            a. assemble the ID (sentenceID + variantID)
            b. check if the keys in the former ones are in the current contrast Variants
                if yes: 
                    then put this in there 
                else:
                    then the assembled ID becomes the new key 
        
        -> ****************** REJECTED ***************************************    
        2. Images are preprocessed as pixel values and go into the dataset       
        -> ****************** REJECTED ***************************************
        
        3. Language preprocessing is done based on the functions declared below  
    
        looks somewhat like.. 
        
             {1,1 : [{about variant1}, {abour variant2}],
              1,2 : [{about variant1}, {about variant2}],
              .....
              } 
              
        the result acquired by extracting values from the dictionary
        """
        for i in range(len(self.lava)): 
            lv = self.lava[i]
            #print(lv)
            id = str(lv['sentenceID']) + ',' + str(lv['variantID'])
            intr = set(self.res.keys()).intersection(set(lv['contrastVariants']))
            #imglink = "images/" + lv['visualFilename'] + ".png"
            
            # Packaging the data
            pack = dict() 
            pack['text'] = lv['text']
            
            try:
                pack['constree'] = lv['syntactic_tree']
            except KeyError:
                continue 
            
            try:
                pack['vision'] = lv['visualFilename']
            except KeyError:
                pack['vision'] = tempdict[lv['equivalentVariants'][0]]['visualFilename']
            
            try:
                pack['sem'] = lv["logicalForm"]
            except KeyError:
                pack['sem'] = tempdict[lv['equivalentVariants'][0]]['logicalForm']
            
            
            if len(set(intr)) != 0:       # 1.b 
                key = list(intr)[0]
                self.res[key].append(pack) 
            else: 
                self.res[id] = [pack] 
                
        data = list(self.res.values())        
        
        # For those with the length 3... 
        two = [i for i in data if len(i) == 2]
        tri = [i for i in data if len(i) == 3] 
        dva = [] 
        for each in tri:
            for i in combinations(each, 2):
                dva.append(list(i))
        # Merge two with dva 
        self.data = two + dva 
        
        
        # Combinations of k 
        self.k = k
        self.comb = list(combinations(self.data, k))        
        
        
    
    """ 
    Data Split 
    70 - 10 - 20 (training, evaluation, testing)
    
    1. shuffle the self.comb
    2. split the self.comb into 70 - 30 
    3. split the 30 into 10 - 20 
    
    """
    def data_splits(self):
        #torch.manual_seed(seed)
        #torch.cuda.manual_seed(seed)
        #torch.cuda.manual_seed_all(seed)
        #torch.backends.cudnn.deterministic = True
        
        # Shuffle the data
        #torch.randperm(len(self.comb))
        shuffled = random.sample(self.comb, len(self.comb))
        # Split the data
        train = shuffled[:int(0.7*len(self.comb))]
        eval = shuffled[int(0.7*len(self.comb)):int(0.8*len(self.comb))]
        test = shuffled[int(0.8*len(self.comb)):]
        
        return train, eval, test



