import torch 
from transformers import CLIPProcessor, CLIPModel, AdamW, get_scheduler
import wandb
import json 
from torch.utils.data import DataLoader
from PIL import Image
from tqdm.auto import tqdm



def main():
    wandb.init() 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    model.load_state_dict(torch.load('cocomodels/COCO_miru_best.pt'))
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    imagepath = "../../../data/LAVA/images/"
    trainpath = "../../../data/LAVA/train.json"
    evalpath = "../../../data/LAVA/val.json"
    
    
    train = json.load(open(trainpath, 'r'))
    val = json.load(open(evalpath, 'r'))
    
    
    batch_size = 8
    epochs = 500
    shuffle = True
    num_workers = 0
    learning_rate = 1e-5
    num_training_steps = epochs * len(train) // batch_size
    
    
    optim = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_scheduler(
        "linear",
        optim,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    
    best_valid_loss = 50000000
    best_train_loss = 50000000
    progress_bar = tqdm(range(num_training_steps))
    
    
    for param in model.vision_model.parameters():
        param.requires_grad = False
        
        
    
    for epoch in range(epochs):
        leng = 0
        cn = 0 
        model.train() 
        tloader = DataLoader(train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        for batch in tloader: 
            text = [] 
            image = [] 
            for x in batch: 
                for y in x: 
                    text += y['constree']
                    imgprocessed = [Image.open(imagepath + y['vision'][z] + '.png') for z in range(len(y['vision']))]
                    image += imgprocessed
                    
            inputs = proc(
                text = text,
                images = image,
                return_tensors = 'pt',
                padding = True
            ).to(device) 
            
            outputs = model(**inputs, return_loss = True)
            loss = outputs.loss
            print('loss: ', loss)
            wandb.log({'COLAVA_miruclip_train_loss': loss})
            loss.backward()
            optim.step()
            scheduler.step()
            optim.zero_grad()
            progress_bar.update(1)
            
        
        
        model.eval() 
        losses = [] 
        vloader = DataLoader(val, batch_size=1, shuffle=shuffle, num_workers=num_workers)
        for batch in vloader: 
            text = [] 
            image = [] 
            for x in batch: 
                for y in x: 
                    text += y['constree']
                    imgprocessed = [Image.open(imagepath + y['vision'][z] + '.png') for z in range(len(y['vision']))]
                    image += imgprocessed
                    
            inputs = proc(
                text = text,
                images = image,
                return_tensors = 'pt',
                padding = True
            ).to(device) 
            
            
            with torch.no_grad():
                outputs = model(**inputs, return_loss = True)
                loss = outputs.loss
                print('validloss: ', loss)
                wandb.log({'COLAVA_miruclip_val_loss': loss})
                losses.append(loss)
                
            if loss < best_valid_loss:
                best_valid_loss = loss
                torch.save(model.state_dict(), 'colavamodels/COLAVA_miruclip_best.pt')
                
                
            sim = outputs.logits_per_text
            
            for i in range(len(sim)): 
                leng += 1 
                res = sim[i].argmax().item()
                if res == i: 
                    cn += 1
        
        
        acc = cn / leng * 100
        wandb.log({'COLAVA_miruclip_val_acc': acc})
        print('acc: ', acc)
        
        
        #var = torch.var(torch.tensor(losses))
        #wandb.log({'K1_miru_clip_val_loss_variance': var})
        
        
    return 


if __name__ == '__main__':
    main()