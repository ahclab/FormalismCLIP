import torch 
from transformers import CLIPProcessor, AdamW, get_scheduler
import wandb 
import json 
from torch.utils.data import DataLoader 
from PIL import Image 
from tqdm.auto import tqdm 
from models import SemGCNCLIP, graphise 
#import hanlp 
from torch_geometric.data import Data 
import re 
import os 
proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SemGCNCLIP(device).to(device)  
# load the model 
model.load_state_dict(torch.load("colavamodels/COLAVA_GCN_best.pt"))
model.eval() 
#han = hanlp.load(hanlp.pretrained.mtl.UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_XLMR_BASE)
tokenizer= model.tokenizer
wordencoder = model.wordencoder.to(device) 

cocopath = '../../../data/COCO/newnewtest.json' 
imagepath = '../../../data/COCO/' 
def is_digit(char):
    try:
        int(char)
        return True 
    except ValueError:
        return False

def cocograph(tok, rel, text_encoder, tokenizer, device):
    #tok = doc['tok'] 
    #rel = doc['sdp/pas']  
    
    
    edge_list = [] 
    edge_index = [] 
    node_index = [] 
    node_list = [] 
    index_transform = dict() 
    
    for i, relations in enumerate(rel): 
        for j, (target, relation) in enumerate(relations):
            if 'det' not in relation and 'aux' not in relation:
                edge_list.append([target - 1, i, relation])
                node_index.append(i) 
                node_index.append(target - 1)
                
    node_index = sorted(list(set(node_index)))  
    node_list = [tok[i] for i in node_index] 
    for i in range(len(node_index)): 
        index_transform[node_index[i]] = i
    
    for i in range(len(edge_list)): 
        edge_list[i][0] = index_transform[edge_list[i][0]]
        edge_list[i][1] = index_transform[edge_list[i][1]]
    
    new_node_num = len(node_index)
    
    for line in edge_list:
        if not is_digit(line[2][-1]):
            continue 
        attr = str(int(line[2][-1]) - 1) 
        node_list.append(attr)  
        edge_index.append([line[0], new_node_num])
        edge_index.append([new_node_num, line[1]])
        new_node_num += 1
    
    if len(node_list) == 0:
        return None, None
        
    x_inp = tokenizer(node_list, padding = True, return_tensors = 'pt').to(device) 
    x_opt = text_encoder(**x_inp) 
    x = x_opt.text_embeds 
    
    edge_index = torch.tensor(edge_index, dtype=torch.long) 
    
    graph = Data(
        x = x,
        edge_index = edge_index.t().contiguous()
    )
        
    
    return graph, node_list 



R_at_1_cnt_T2I = 0
R_at_1_cnt_I2T = 0 
R_at_5_cnt_T2I = 0
R_at_5_cnt_I2T = 0
R_at_10_cnt_T2I = 0
R_at_10_cnt_I2T = 0

image = []  
text = [] 
graph = [] 

test = json.load(open(cocopath, 'r')) 
test = test
#mondaiji = []

with torch.no_grad(): 
    for i in tqdm(range(len(test))):
        txt = test[i]['sentence']
    #hanout = han(txt) 
        tok = test[i]['tok']
        rel = test[i]['rel']
        graphised = cocograph(tok, rel, wordencoder, tokenizer, device) 
        if graphised[0] == None: 
            continue
        graph.append(graphised)  
        image.append(Image.open(os.path.join(imagepath, test[i]['path'])))
        text.append(txt) 


     
    pix_val = proc(images=image, return_tensors="pt", padding=True)['pixel_values'].to(device)
            


    outputs = model(
        text = text,
        graphs = graph, 
        pixel_values = pix_val, 
        return_loss = False,
        edge_weight=None,  
    ) 


Ks = [1, 5, 10]
max_K = max(Ks)
n_count = {K : 0 for K in Ks} 
n_total = 0 

T2I = outputs['logits_per_graph'] 
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
print('R@5 T2I: ', R_at_5_cnt_T2I / len(T2I))
print('R@10 T2I: ', R_at_10_cnt_T2I / len(T2I))
print("-------------------------------")
print('R@1 I2T: ', R_at_1_cnt_I2T / len(I2T))
print('R@5 I2T: ', R_at_5_cnt_I2T / len(I2T))
print('R@10 I2T: ', R_at_10_cnt_I2T / len(I2T))


"""
loader = DataLoader(test, batch_size=16, shuffle=True, num_workers=0)
leng = 0 
cn = 0 
for batch in loader: 
    text = batch['sentence']
    graph = [] 
    mondaiji = []
    image = [Image.open(os.path.join(imagepath, i)) for i in batch['path']] 
    for i in range(len(text)): 
        t = text[i]
        #tok = batch['tok'][i]
        #rel = batch['rel'][i]
        hanout = han(t)
        graphised= cocograph(hanout, wordencoder, tokenizer, device)
        if graphised[0] == None:
            mondaiji.append(i)
            continue
        graph.append(graphised)
        
    for i in mondaiji:
        text.pop(i)
        image.pop(i)
        
    pix_val = proc(images=image, return_tensors="pt", padding=True)['pixel_values'].to(device)
    
    with torch.no_grad(): 
        outputs = model(
            text = text,
            graphs = graph, 
            pixel_values = pix_val, 
            return_loss = False,
            edge_weight=None,
        )
        
    sim = outputs['logits_per_graph']
    for i in range(len(sim)):
        leng += 1 
        res = sim[i].argmax().item()
        if res == i: 
            cn += 1
        
acc = cn / leng * 100
print('acc: ', acc)
"""