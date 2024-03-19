import json 
import torch 
from models import SemGCNCLIP, graphise  
from transformers import CLIPProcessor 
from PIL import Image 
from torch.utils.data import DataLoader 
imagepath = "../../../data/LAVA/images/"
trainpath = "../../../data/LAVA/train.json"
evalpath = "../../../data/LAVA/val.json"
recalltestpath = "doubletest.json"
acctestpath = "../../../data/LAVA/double_tst.json"

train = json.load(open(trainpath, 'r'))
val = json.load(open(evalpath, 'r'))
recalltest = json.load(open(recalltestpath, 'r')) 
acctest = json.load(open(acctestpath, 'r'))

proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
model = SemGCNCLIP(device).to(device)  
# load the model 
model.load_state_dict(torch.load("lavamodels/K2_wtxt_graphconv_best.pt"))



R_at_1_cnt_T2I = 0
R_at_1_cnt_I2T = 0 
R_at_5_cnt_T2I = 0
R_at_5_cnt_I2T = 0
R_at_10_cnt_T2I = 0
R_at_10_cnt_I2T = 0

image = []  
text = [] 
graph = [] 


for i in range(len(recalltest)): 
    targ = recalltest[i]
    text.append(targ['text'])
    pyg = [graphise(targ['sem'], model.wordencoder, model.tokenizer, device)]
    graph += pyg 
    imgproc = [Image.open(imagepath + targ['vision'] + ".png")]
    image += imgproc 
        
pix_val = proc(images=image, return_tensors="pt", padding=True)['pixel_values'].to(device)
            

outputs = model(
    text = text,
    graphs = graph, 
    pixel_values = pix_val, 
    return_loss = False,
    edge_weight=None,  
) 

T2I = outputs['logits_per_graph'] 
I2T = outputs['logits_per_image'] 

# Recall at 1 
for i in range(len(T2I)):
    if T2I[i][i] == max(T2I[i]):
        R_at_1_cnt_T2I += 1 
        
for i in range(len(I2T)):
    if I2T[i][i] == max(I2T[i]):
        R_at_1_cnt_I2T += 1 
        
        
# Recall at 5
for i in range(len(T2I)):
    if T2I[i][i] in sorted(T2I[i], reverse=True)[:5]:
        R_at_5_cnt_T2I += 1 
            
        
for i in range(len(I2T)):
    if I2T[i][i] in sorted(I2T[i], reverse=True)[:5]:
        R_at_5_cnt_I2T += 1 
            
        
# Recall at 10
for i in range(len(T2I)):
    if T2I[i][i] in sorted(T2I[i], reverse=True)[:10]:
        R_at_10_cnt_T2I += 1 
            
        
for i in range(len(I2T)):
    if I2T[i][i] in sorted(I2T[i], reverse=True)[:10]:
        R_at_10_cnt_I2T += 1 
            
        
print('R@1 T2I: ', R_at_1_cnt_T2I / len(T2I))
print('R@1 I2T: ', R_at_1_cnt_I2T / len(I2T))
print('R@5 T2I: ', R_at_5_cnt_T2I / len(T2I))
print('R@5 I2T: ', R_at_5_cnt_I2T / len(I2T))
print('R@10 T2I: ', R_at_10_cnt_T2I / len(T2I))
print('R@10 I2T: ', R_at_10_cnt_I2T / len(I2T))



loader = DataLoader(acctest, batch_size=1, shuffle=True, num_workers=0)
leng = 0 
cn = 0 
for batch in loader: 
    text = [] 
    graph = [] 
    image = [] 
    for x in batch: 
        for y in x[0]:
            text += y['text'] 
            pyg = [graphise(sem, model.wordencoder, model.tokenizer, device) for sem in y['sem']]
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