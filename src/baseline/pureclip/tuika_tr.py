import torch 
from transformers import CLIPModel, CLIPProcessor, AdamW, get_scheduler
import wandb 
import json 
from torch.utils.data import DataLoader 
from PIL import Image 
from tqdm.auto import tqdm 


from torch_geometric.data import Data 
import re 
import os 




def main():
    wandb.init() 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Datapaths 
    coco_image = '../../../data/COCO/'
    coco_lang  = '../../../data/COCO/dataset_coco.json'
    lava_image = '../../../data/LAVA/images/' 
    lava_lang_train = '../../../data/LAVA/train.json'
    lava_lang_eval = '../../../data/LAVA/val.json'
    
    # Data 
    coco = json.load(open(coco_lang, 'r'))
    coco = coco['images']
    coco_train = [i for i in coco if i['split'] == 'train']
    coco_val = [i for i in coco if i['split'] == 'val']
    coco_train = [{'path' : os.path.join(i['filepath'], i['filename']),
              'sentence' : re.sub(r'[^\w\s]', '', i['sentences'][0]['raw'])} for i in coco_train]
    coco_val = [{'path' : os.path.join(i['filepath'], i['filename']),
              'sentence' : re.sub(r'[^\w\s]', '', i['sentences'][0]['raw'])} for i in coco_val]
    
    lava_train = json.load(open(lava_lang_train, 'r'))
    lava_val = json.load(open(lava_lang_eval, 'r'))
    
    
    # Hyperparams 
    coco_train_batch_size = 32
    coco_eval_batch_size = 32
    lava_train_batch_size = 8 
    lava_eval_batch_size = 8 
    epochs = 1000 
    shuffle = True 
    num_workers = 0
    learning_rate = 1e-5
    coco_num_training_steps = epochs * len(coco_train) // coco_train_batch_size
    lava_num_training_steps = epochs * len(lava_train) // lava_train_batch_size
    
    
    # Optimiser and Scheduler
    optim = AdamW(model.parameters(), lr=learning_rate) 
    scheduler = get_scheduler(
        "linear",
        optim,
        num_warmup_steps=0,
        num_training_steps=coco_num_training_steps + lava_num_training_steps
    )
    
    best_coco_valid_loss = 50000000
    best_lava_train_loss = 50000000
    best_coco_train_loss = 50000000
    
    progress_bar = tqdm(range(coco_num_training_steps)) 
    
    # Freeze the vision model
    for param in model.vision_model.parameters():
        param.requires_grad = False
    
    for epoch in range(epochs):
        leng = 0 
        cn = 0 
        
        # Training 
        model.train()
        coco_tloader = DataLoader(
            coco_train,
            batch_size=coco_train_batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
        
        lava_tloader = DataLoader(
            lava_train,
            batch_size=lava_train_batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
        
        for batch in coco_tloader:
            text = batch['sentence'] 
            image = [Image.open(os.path.join(coco_image, i)) for i in batch['path']] 
            
            inputs = proc(
                text = text,
                images = image,
                return_tensors="pt",
                padding=True
            ).to(device) 
            
            outputs = model(**inputs, return_loss=True)
            loss = outputs.loss
            print('coco_loss: ', loss)
            wandb.log({'Tuika_COCO_puretrain_loss': loss})
            loss.backward()
            optim.step() 
            scheduler.step()
            optim.zero_grad()
            progress_bar.update(1)
            
            if loss < best_coco_train_loss:
                best_coco_train_loss = loss
        
        for batch in lava_tloader:
            text = [] 
            image = [] 
            for x in batch: 
                for y in x: 
                    text += y['text'] 
                    imgprocessed = [Image.open(lava_image + y['vision'][z] + '.png') for z in range(len(y['vision']))]
                    image += imgprocessed 
            
            inputs = proc(
                text = text,
                images = image,
                return_tensors = 'pt',
                padding = True
            ).to(device)
            
            outputs = model(**inputs, return_loss=True)
            loss = outputs.loss 
            print('lava_loss: ', loss)
            wandb.log({'Tuika_LAVA_puretrain_loss': loss})
            loss.backward()
            optim.step()
            scheduler.step()
            optim.zero_grad()
            progress_bar.update(1)
            
            if loss < best_lava_train_loss:
                best_lava_train_loss = loss
                
        # Evaluation
        model.eval()
        losses = [] 
        coco_vloader = DataLoader(
            coco_val,
            batch_size=coco_eval_batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
        
        for batch in coco_vloader: 
            text = batch['sentence'] 
            image = [Image.open(os.path.join(coco_image, i)) for i in batch['path']] 
            
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
                print('coco_validloss: ', validloss)
                wandb.log({'Tuika_COCO_pureclip_val_loss': validloss})
                
                if validloss < best_coco_valid_loss:
                    best_coco_valid_loss = validloss
                    torch.save(model.state_dict(), 'tuikamodels/Tuika_COCO_pureclip_best.pt')
                    print("Model Saved") 
                    
            sim = outputs.logits_per_text
            
            for i in range(len(sim)):
                leng += 1
                res = sim[i].argmax().item() 
                if res == i: 
                    cn += 1 
            
        acc = cn / leng * 100
        print('COCO accuracy: ', acc)
        wandb.log({'Tuika_COCO_pureclip_accuracy': acc})
        
        var = torch.var(torch.tensor(losses))
        wandb.log({'Tuika_COCO_pureclip_variance': var}) 
        
    
    return 









if __name__ == '__main__':
    main()