import torch 
from transformers import CLIPProcessor, AdamW, get_scheduler
import wandb 
import json 
from torch.utils.data import DataLoader 
from PIL import Image 
from tqdm.auto import tqdm 
from models import * 
from torch_geometric.data import Data 
import re 
import os 
import sys 
from nltk import Tree 
sys.path.append("../Core")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GuiGCLIP(device).to(device)
model.load_state_dict(torch.load("lavamodels/GuiG_LAVA_best.pt"))
model.eval() 
proc = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    

imagepath = "../../../../../data/LAVA/images/"
trainpath = "../../../../../data/LAVA/train.json"
evalpath = "../../../../../data/LAVA/val.json"
testpath = "../../../../../data/LAVA/tst.json"
test = json.load(open(testpath, 'r')) 
dict_dir = os.path.join('../data/train', 'para_train_dict.pt') 
w2i_dict = torch.load(dict_dir) 
syntax_dict = w2i_dict['syntax']
level_dict = w2i_dict['level']

R_at_1_cnt_T2I = 0
R_at_1_cnt_I2T = 0 
R_at_5_cnt_T2I = 0
R_at_5_cnt_I2T = 0
R_at_10_cnt_T2I = 0
R_at_10_cnt_I2T = 0

image = []  
text = [] 
tree = [] 


with torch.no_grad():
    for i in range(len(test)):
        for j in range(len(test[i][0])): 
            targ = test[i][0] 
            text.append(targ[j]['text'])
            tree.append(targ[j]['constree'])
        #pyg = [graphise(targ[j]['sem'], model.wordencoder, model.tokenizer, device)]
        #graph += pyg 
            imgproc = [Image.open(imagepath + targ[j]['vision'] + ".png")]
            image += imgproc 
        
    pix_val = proc(images=image, return_tensors="pt", padding=True)['pixel_values'].to(device)
            
    tree_nltk = [Tree.fromstring(t) for t in tree] 




    discard = [] 
            #1. syn_seq 
    syn_seqs = [POS_for_inputs(t, syntax_dict) for t in tree]
    for i in range(len(syn_seqs)):
        if len(syn_seqs[i]) > 51: 
            discard.append(i)
             
            
            
            
            #2. lvl_seq 
    lvl_seqs = [level_sequence_for_input(t, level_dict) for t in tree_nltk]
    for i in range(len(lvl_seqs)):
        if len(lvl_seqs[i]) > 51: 
            discard.append(i)
            
            
            
            #3. pos_seq
    poss = [POS_extractor(t) for t in tree] 
    pos_seqs = [position_generator(t) for t in poss]
    for i in range(len(pos_seqs)):
        if len(pos_seqs[i]) > 51: 
            discard.append(i) 
            
            
            #4. path_mask
    numbered = [number_tagger(t) for t in tree] 
    numbered_t = [Tree.fromstring(t) for t in numbered] 
    paths = [get_paths_to_leaves(t) for t in numbered_t]
    numbered_pos = [POS_extractora(t) for t in numbered] 
    path_masks = [
        path_index_generator(path, pos) 
        for path, pos in zip(paths, numbered_pos) 
    ]
    for i in range(len(path_masks)):
        if len(path_masks[i]) > 51: 
            discard.append(i)
                    
                    
    discard = list(set(discard)) 
    syn_seqs = [syn_seqs[i] for i in range(len(syn_seqs)) if i not in discard]
    lvl_seqs = [lvl_seqs[i] for i in range(len(lvl_seqs)) if i not in discard]
    pos_seqs = [pos_seqs[i] for i in range(len(pos_seqs)) if i not in discard]
    path_masks = [path_masks[i] for i in range(len(path_masks)) if i not in discard]
    text = [text[i] for i in range(len(text)) if i not in discard]
    image = [image[i] for i in range(len(image)) if i not in discard]
            
    syn_seqs = torch.tensor(syn_seqs).to(device)
    lvl_seqs = torch.tensor(lvl_seqs).to(device)
    pos_seqs = torch.tensor(pos_seqs).to(device)
    path_masks = torch.tensor(path_masks).to(device) 
    pix_val = proc(
        images=image,
        return_tensors='pt',
        padding = True 
    )['pixel_values'].to(device)
            
    outputs = model(
        texts = text, 
        syn_seqs = syn_seqs,
        lvl_seqs = lvl_seqs,
        pos_seqs = pos_seqs,
        path_masks = path_masks,
        pixel_values = pix_val,
        return_loss = True
    )


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



loader = DataLoader(test, batch_size=1, shuffle=True, num_workers=0)
leng = 0 
cn = 0 
for batch in loader: 
    text = [] 
    tree = [] 
    image = [] 
    for x in batch: 
        for y in x:
            text += y['text']
            tree += y['constree'] 
            #pyg = [graphise(sem, model.wordencoder, model.tokenizer, device) for sem in y['sem']]
            #graph += pyg
            imgproccesed = [Image.open(imagepath + y['vision'][z] + '.png') for z in range(len(y['vision']))]
            image += imgproccesed 
            
    tree_nltk = [Tree.fromstring(t) for t in tree]
            
    """
            syn_seq 
            lvl_seq 
            pos_seq
            path_mask 
    """
            
    discard = [] 
            #1. syn_seq 
    syn_seqs = [POS_for_inputs(t, syntax_dict) for t in tree]
    for i in range(len(syn_seqs)):
        if len(syn_seqs[i]) > 51: 
            discard.append(i)
             
            
            
            
            #2. lvl_seq 
    lvl_seqs = [level_sequence_for_input(t, level_dict) for t in tree_nltk]
    for i in range(len(lvl_seqs)):
        if len(lvl_seqs[i]) > 51: 
            discard.append(i)
            
            
            
            #3. pos_seq
    poss = [POS_extractor(t) for t in tree] 
    pos_seqs = [position_generator(t) for t in poss]
    for i in range(len(pos_seqs)):
        if len(pos_seqs[i]) > 51: 
            discard.append(i) 
            
            
            #4. path_mask
    numbered = [number_tagger(t) for t in tree] 
    numbered_t = [Tree.fromstring(t) for t in numbered] 
    paths = [get_paths_to_leaves(t) for t in numbered_t]
    numbered_pos = [POS_extractora(t) for t in numbered] 
    path_masks = [
        path_index_generator(path, pos) 
        for path, pos in zip(paths, numbered_pos) 
    ]
    for i in range(len(path_masks)):
        if len(path_masks[i]) > 51: 
            discard.append(i)
                    
                    
    discard = list(set(discard)) 
    syn_seqs = [syn_seqs[i] for i in range(len(syn_seqs)) if i not in discard]
    lvl_seqs = [lvl_seqs[i] for i in range(len(lvl_seqs)) if i not in discard]
    pos_seqs = [pos_seqs[i] for i in range(len(pos_seqs)) if i not in discard]
    path_masks = [path_masks[i] for i in range(len(path_masks)) if i not in discard]
    text = [text[i] for i in range(len(text)) if i not in discard]
    image = [image[i] for i in range(len(image)) if i not in discard]
            
    syn_seqs = torch.tensor(syn_seqs).to(device)
    lvl_seqs = torch.tensor(lvl_seqs).to(device)
    pos_seqs = torch.tensor(pos_seqs).to(device)
    path_masks = torch.tensor(path_masks).to(device) 
    pix_val = proc(
        images=image,
        return_tensors='pt',
        padding = True 
    )['pixel_values'].to(device)
            
    outputs = model(
        texts = text, 
        syn_seqs = syn_seqs,
        lvl_seqs = lvl_seqs,
        pos_seqs = pos_seqs,
        path_masks = path_masks,
        pixel_values = pix_val,
        return_loss = True
    )
        
    sim = outputs['logits_per_text']
    for i in range(len(sim)):
        leng += 1 
        res = sim[i].argmax().item()
        if res == i: 
            cn += 1
        
acc = cn / leng * 100
print('acc: ', acc)