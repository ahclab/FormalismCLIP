import torch
from models import TG_CLIP, modify_syntax_tree, MASK 
import wandb
import json
from torch.utils.data import DataLoader
from PIL import Image
from tqdm.auto import tqdm
import os
from transformers import CLIPProcessor, AdamW, get_scheduler




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TG_CLIP(device).to(device)
#model.load_state_dict(torch.load("cocomodels/TG_COCO_best.pt"))
model.load_state_dict(torch.load("colavamodels/TG_COLAVA_best.pt"))
model.eval() 
proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


imagepath = "../../../data/COCO/" 
trainpath = "../../../data/LAVA/train.json"
evalpath = "../../../data/LAVA/val.json"
testpath = "../../../data/COCO/newtest.json"

with torch.no_grad():
    test = json.load(open(testpath, 'r')) 
    test = [elem for elem in test if modify_syntax_tree(elem['tree']) != None]
    test = [elem for elem in test
            if proc(
                text=modify_syntax_tree(elem['tree']),
                return_tensors='pt')['input_ids'].shape[1] <= 77]

    # newtest : those in test which have no error with the modify_syntax_tree into MASK 
     
    mondai = [] 
    for i in tqdm(range(len(test))):
        tree = modify_syntax_tree(test[i]['tree'])
        try:
            mask = MASK(tree)
        except IndexError:
            print("IndexError") 
            mondai.append(i)
        
    print("mondai len : ", len(mondai)) 
    test = [test[i] for i in range(len(test)) if i not in mondai]



    image = []  
    #text = [] 
    tree = [] 


    for i in tqdm(range(len(test))):
        
            #text.append(targ[j]['text'])
        tree.append(modify_syntax_tree(test[i]['tree'])) 
        #pyg = [graphise(targ[j]['sem'], model.wordencoder, model.tokenizer, device)]
        #graph += pyg 
        image.append(Image.open(os.path.join(imagepath, test[i]['path'])))
        
        
    pix_val = proc(
        images=image,
        return_tensors="pt",
        padding=True)['pixel_values'].to(device)
    #tree = [modify_syntax_tree(t) for t in tree]
    


    outputs = model(
        tree = tree,
        pixel_values = pix_val,
        return_loss = False,
    ) 


R_at_1_cnt_T2I = 0
R_at_1_cnt_I2T = 0 
R_at_5_cnt_T2I = 0
R_at_5_cnt_I2T = 0
R_at_10_cnt_T2I = 0
R_at_10_cnt_I2T = 0



Ks = [1, 5, 10]
max_K = max(Ks)
n_count = {K : 0 for K in Ks} 
n_total = 0 

T2I = outputs['logits_per_text'] 
I2T = outputs['logits_per_image'] 

T2I_ordered_index = torch.argsort(T2I, dim=1, descending=True)
I2T_ordered_index = torch.argsort(I2T, dim=1, descending=True) 



# R@1 T2I 
R_at_1_T2I_extract = T2I_ordered_index[:, :1] 
for i in range(len(R_at_1_T2I_extract)):
    if i in R_at_1_T2I_extract[i]: 
        R_at_1_cnt_T2I += 1
        
# R@1 I2T
R_at_1_I2T_extract = I2T_ordered_index[:, :1]
for i in range(len(R_at_1_I2T_extract)):
    if i in R_at_1_I2T_extract[i]: 
        R_at_1_cnt_I2T += 1
        
# R@5 T2I
R_at_5_T2I_extract = T2I_ordered_index[:, :5]
for i in range(len(R_at_5_T2I_extract)):
    if i in R_at_5_T2I_extract[i]: 
        R_at_5_cnt_T2I += 1
        
# R@5 I2T
R_at_5_I2T_extract = I2T_ordered_index[:, :5]
for i in range(len(R_at_5_I2T_extract)):
    if i in R_at_5_I2T_extract[i]: 
        R_at_5_cnt_I2T += 1
        
# R@10 T2I
R_at_10_T2I_extract = T2I_ordered_index[:, :10]
for i in range(len(R_at_10_T2I_extract)):
    if i in R_at_10_T2I_extract[i]: 
        R_at_10_cnt_T2I += 1
        
# R@10 I2T
R_at_10_I2T_extract = I2T_ordered_index[:, :10]
for i in range(len(R_at_10_I2T_extract)):
    if i in R_at_10_I2T_extract[i]: 
        R_at_10_cnt_I2T += 1


            
        
print('R@1 T2I: ', R_at_1_cnt_T2I / len(T2I))
print('R@1 I2T: ', R_at_1_cnt_I2T / len(I2T))
print('R@5 T2I: ', R_at_5_cnt_T2I / len(T2I))
print('R@5 I2T: ', R_at_5_cnt_I2T / len(I2T))
print('R@10 T2I: ', R_at_10_cnt_T2I / len(T2I))
print('R@10 I2T: ', R_at_10_cnt_I2T / len(I2T))
"""

loader = DataLoader(test, batch_size=32, shuffle=True, num_workers=0)
leng = 0 
cn = 0 


for batch in loader: 
    text = batch['tree']
    text = [modify_syntax_tree(t) for t in text]
    image = [Image.open(os.path.join(imagepath, i)) for i in batch['path']]
    pix_val = proc(
        images=image,
        return_tensors="pt",
        padding=True)['pixel_values'].to(device)
    
  
    with torch.no_grad():
        outputs = model(
            tree = text,
            pixel_values = pix_val,
            return_loss = False,
        )
        
    sim = outputs['logits_per_text'] 
    for i in range(len(sim)):
        leng += 1 
        res = sim[i].argmax().item()
        if res == i: 
            cn += 1 
        
acc = cn / leng * 100
print("acc: ", acc) 

"""