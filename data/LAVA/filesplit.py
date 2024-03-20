from lava import LAVA 
import json 

"""
Splits the data and creates train.json, val.json, and tst.json
"""

k = int(input("Value of K : ")) 
lv = LAVA(k) 


train, eval, test = lv.data_splits() 


train_path = "train.json"
val_path = "val.json" 
tst_path = "tst.json" 

with open(train_path, 'w') as f: 
  json.dump(train, f) 

with open(val_path, 'w') as f: 
  json.dump(eval, f) 

with open(tst_path, 'w') as f: 
  json.dump(test, f) 
