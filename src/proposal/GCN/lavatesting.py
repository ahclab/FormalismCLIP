import json 
import torch 
from models import SemGCNCLIP, graphise, graphise_MK2
from transformers import CLIPProcessor 
from PIL import Image 
from torch.utils.data import DataLoader 
imagepath = "../../../data/LAVA/images/"
trainpath = "../../../data/LAVA/train.json"
evalpath = "../../../data/LAVA/val.json"
testpath = "../../../data/LAVA/tst.json"

train = json.load(open(trainpath, 'r'))
val = json.load(open(evalpath, 'r'))
test = json.load(open(testpath, 'r')) 

proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model = SemGCNCLIP(device).to(device)  
# load the model 
model.load_state_dict(torch.load("MK2_lavamodels/MK2_graphconv_best.pt"))

model.eval() 

R_at_1_cnt_T2I = 0
R_at_1_cnt_I2T = 0 
R_at_5_cnt_T2I = 0
R_at_5_cnt_I2T = 0
R_at_10_cnt_T2I = 0
R_at_10_cnt_I2T = 0

image = []  
text = [] 
graph = [] 

for i in range(len(test)):
    for j in range(len(test[i][0])): 
        targ = test[i][0] 
        text.append(targ[j]['text'])
        pyg = [graphise_MK2(targ[j]['sem'], model.wordencoder, model.tokenizer, device)]
        graph += pyg 
        imgproc = [Image.open(imagepath + targ[j]['vision'] + ".png")]
        image += imgproc 
        
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



loader = DataLoader(test, batch_size=1, shuffle=True, num_workers=0)
leng = 0 
cn = 0 
for batch in loader: 
    text = [] 
    graph = [] 
    image = [] 
    for x in batch: 
        for y in x:
            text += y['text'] 
            pyg = [graphise_MK2(sem, model.wordencoder, model.tokenizer, device) for sem in y['sem']]
            graph += pyg
            imgproccesed = [Image.open(imagepath + y['vision'][z] + '.png') for z in range(len(y['vision']))]
            image += imgproccesed 
    
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