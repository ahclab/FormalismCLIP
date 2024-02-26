import torch 
from transformers import CLIPProcessor, CLIPModel, AdamW, get_scheduler  
import wandb 
import os
import json
from torch.utils.data import DataLoader
from PIL import Image
from tqdm.auto import tqdm







def main():
    #wandb.login()
    #wandb.init(project='lava')
    #print("Initiating wandb")
    wandb.init()
    #print("Initiation complete")
    
    # set the gpu device as cuda number 2
    #device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')  
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    model.load_state_dict(torch.load('cocomodels/COCO_GCN_best.pt'))
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    #print("models uploaded") 
    
    #imagepath = "../../../data/LAVA/images/"
    
    imagepath = "../../../data/LAVA/images/"
    trainpath = "../../../data/LAVA/train.json"
    evalpath = "../../../data/LAVA/val.json"
    testpath = "../../../data/LAVA/tst.json"
    
    
    train = json.load(open(trainpath, 'r'))
    val = json.load(open(evalpath, 'r'))
    test = json.load(open(testpath, 'r')) 
    # hyperparams 
    batch_size = 8   ## not sure 
    epochs = 500
    shuffle = True
    num_workers = 0 
    learning_rate = 1e-5 
    num_training_steps = epochs * len(train) // batch_size
    ## maybe have to modifiy the upper line 
    
    # Optimiser and Scheduler 
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
    
    # Freeze the vision model 
    for param in model.vision_model.parameters():
        param.requires_grad = False
        
    
    # Training/Eval Loop
    for epoch in range(epochs):
        # Training
        leng = 0 
        cn = 0
        model.train() 
        tloader = DataLoader(train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        for batch in tloader: 
            text = [] 
            image = [] 
            for x in batch: 
                for y in x: 
                    text += y['text'] 
                    imgprocessed = [Image.open(imagepath + y['vision'][z] + '.png') for z in range(len(y['vision']))]
                    image += imgprocessed 
                
            inputs = proc(
                text = text,
                images = image,
                return_tensors = 'pt',
                padding = True
            ).to(device)
            
            outputs = model(**inputs, return_loss=True)
            loss = outputs.loss 
            print('loss: ', loss)
            wandb.log({'COLAVA_puretrain_loss': loss})
            loss.backward()
            optim.step()
            scheduler.step()
            optim.zero_grad()
            progress_bar.update(1)
            
            if loss < best_train_loss:
                best_train_loss = loss 
                
        # Evaluation 
        model.eval() 
        losses = [] 
        vloader = DataLoader(val, batch_size=1, shuffle=shuffle, num_workers=num_workers)
        for batch in vloader: 
            text = [] 
            image = [] 
            for x in batch: 
                for y in x: 
                    text += y['text'] 
                    imgprocessed = [Image.open(imagepath + y['vision'][z] + '.png') for z in range(len(y['vision']))]
                    image += imgprocessed 
            inputs = proc(
                text = text,
                images = image,
                return_tensors = 'pt',
                padding = True
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs, return_loss=True)
                validloss = outputs.loss
                losses.append(validloss) 
                print('validloss: ', validloss)
                wandb.log({'COLAVA_pureclip_val_loss': validloss})
                
                if validloss < best_valid_loss:
                    best_valid_loss = validloss 
                    torch.save(model.state_dict(), 'colavamodels/COLAVA_pureclip_best.pt')
                    
            sim = outputs.logits_per_text
                 
            for i in range(len(sim)): 
                leng += 1 
                res = sim[i].argmax().item()
                if res == i:
                    cn += 1
        acc = cn / leng * 100 
        wandb.log({'COLAVA_pureclip_acc': acc})
        print('acc: ', acc)
    
    # T 
    
        #torch.save(model.state_dict(), 'models/K1_only_clip_' + str(epoch + 1) + '.pt')
        #var = torch.var(torch.tensor(losses))      
        #wandb.log({'K1_only_clip_val_var': var}) 
             
    
    return 















if __name__ == '__main__':
    main()