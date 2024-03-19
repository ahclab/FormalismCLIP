import json 
import torch 
  
from transformers import CLIPProcessor, CLIPModel, AdamW, get_scheduler   
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
# load the model 

model.load_state_dict(torch.load("lavamodels/K1_miru_clip_best.pt", map_location='cuda:0'))
model.eval()


R_at_1_cnt_T2I = 0
R_at_1_cnt_I2T = 0 
R_at_5_cnt_T2I = 0
R_at_5_cnt_I2T = 0
R_at_10_cnt_T2I = 0
R_at_10_cnt_I2T = 0

image = []  
text = [] 
#graph = [] 

for i in range(len(test)):
    for j in range(len(test[i][0])): 
        targ = test[i][0] 
        text.append(targ[j]['constree'])
        #graph.append(targ[j]['sem']) 
        imgproc = [Image.open(imagepath + targ[j]['vision'] + ".png")]
        image += imgproc 
        
#pix_val = proc(images=image, return_tensors="pt", padding=True)['pixel_values'].to(device)
            


inputs = proc(
    text = text,
    images = image,
    return_tensors = "pt",
    padding = True
).to(device) 

outputs = model(**inputs, return_loss = False)

T2I = outputs.logits_per_text 
I2T = outputs.logits_per_image 


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
    #graph = [] 
    image = [] 
    for x in batch: 
        for y in x:
            text += y['constree'] 
            #graph += y['sem']
            imgproccesed = [Image.open(imagepath + y['vision'][z] + '.png') for z in range(len(y['vision']))]
            image += imgproccesed 
    
    #pix_val = proc(images=image, return_tensors="pt", padding=True)['pixel_values'].to(device)
    inputs = proc(
        text= text, 
        images= image, 
        return_tensors= "pt",
        padding = True
    ).to(device)
    with torch.no_grad(): 
        outputs = model(**inputs, return_loss = False)
        
    sim = outputs.logits_per_text
    for i in range(len(sim)):
        leng += 1 
        res = sim[i].argmax().item()
        if res == i: 
            cn += 1
        
acc = cn / leng * 100
print('acc: ', acc)