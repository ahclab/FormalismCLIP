import torch
from models import TG_CLIP, modify_syntax_tree
import wandb
import json
from torch.utils.data import DataLoader
from PIL import Image
from tqdm.auto import tqdm
import os
from transformers import CLIPProcessor, AdamW, get_scheduler

def main():
    wandb.init()
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = TG_CLIP(device).to(device)
    model.load_state_dict(torch.load("cocomodels/TG_COCO_best.pt"))
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    #trainpath = '../../../data/COCO/newtrain.json'
    #validpath = '../../../data/COCO/newval.json'
    #imagepath = '../../../data/COCO/'
    #newtrain = json.load(open(trainpath, 'r'))
    #newval = json.load(open(validpath, 'r'))
    # If there isn’t a single small character inside the tree, remove it
    #newtrain = [elem for elem in newtrain if modify_syntax_tree(elem['tree']) != None]
    #newval = [elem for elem in newval if modify_syntax_tree(elem['tree']) != None]
    #newtrain = [elem for elem in newtrain
    #            if proc(
    #                text=modify_syntax_tree(elem['tree']),
    #                return_tensors='pt')['input_ids'].shape[1] <= 77]
    #newval = [elem for elem in newval
    #          if proc(
    #              text=modify_syntax_tree(elem['tree']),
    #              return_tensors='pt')['input_ids'].shape[1] <= 77]
    
    imagepath = "../../../data/LAVA/images/" 
    trainpath = "../../../data/LAVA/train.json"
    evalpath = "../../../data/LAVA/val.json"
    
    
    train = json.load(open(trainpath, 'r'))
    val = json.load(open(evalpath, 'r'))
    
    #Hyperparams
    train_batch_size = 8
    eval_batch_size = 1
    epochs = 1000
    shuffle = True
    num_workers = 0
    learning_rate = 1e-5
    num_training_steps = epochs * len(train) // train_batch_size
    optim = AdamW(model.parameters(), lr = learning_rate)
    scheduler = get_scheduler(
        'linear',
        optim,
        num_warmup_steps = 0,
        num_training_steps = num_training_steps
    )
    best_valid_loss = 5000000
    best_train_loss = 5000000
    progress_bar = tqdm(range(num_training_steps))
    # Freeze the vision model
    for param in model.vision_model.parameters():
        param.requires_grad = False
    for epoch in range(epochs):
        leng = 0
        cn = 0
        # Training
        model.train()
        tloader = DataLoader(
            train,
            batch_size = train_batch_size,
            shuffle = shuffle,
            num_workers = num_workers
        )
        for batch in tloader:
            tree = [] 
            image = [] 
            for x in batch:
                for y in x:
                    tree += y['constree'] 
                    imgprocessed = [Image.open(imagepath + y['vision'][z] + '.png') for z in range(len(y['vision']))]
                    image += imgprocessed
                    
            pix_val = proc(
                images = image,
                return_tensors='pt',
                padding = True
            )['pixel_values'].to(device)
            tree = [modify_syntax_tree(elem) for elem in tree]
            #text = batch['tree']
            #text = [modify_syntax_tree(elem) for elem in text]
            #image = [Image.open(os.path.join(imagepath, i)) for i in batch['path']]
            
            
            #pix_val = proc(
            #    images = image,
            #    return_tensors='pt',
            #    padding = True
            #)['pixel_values'].to(device)
            
            outputs = model(
                tree = tree,
                pixel_values = pix_val,
                return_loss = True
            )
            loss = outputs['loss']
            #print(“Did we make it here?“)
            #print(“loss : “, loss)
            wandb.log({'TG_COLAVA_loss': loss})
            loss.backward()
            optim.step()
            scheduler.step()
            optim.zero_grad()
            progress_bar.update(1)
            if loss < best_train_loss:
                best_train_loss = loss
        # Validation
        model.eval()
        losses = []
        vloader = DataLoader(
            val,
            batch_size = eval_batch_size,
            shuffle = shuffle,
            num_workers = num_workers
        )
        for batch in vloader:
            tree = [] 
            image = [] 
            for x in batch:
                for y in x:
                    tree += y['constree'] 
                    imgprocessed = [Image.open(imagepath + y['vision'][z] + '.png') for z in range(len(y['vision']))]
                    image += imgprocessed
            pix_val = proc(
                images = image,
                return_tensors='pt',
                padding = True
            )['pixel_values'].to(device)
            tree = [modify_syntax_tree(elem) for elem in tree]
            with torch.no_grad():
                try:
                    outputs = model(
                        tree = tree,
                        pixel_values = pix_val,
                        return_loss = True
                    )
                except RuntimeError: 
                    continue 
                validloss = outputs['loss']
                #print(“validloss : “, validloss)
                losses.append(validloss)
                wandb.log({'TG_COLAVA_validloss': validloss})
                if validloss < best_valid_loss:
                    best_valid_loss = validloss
                    torch.save(model.state_dict(), 'colavamodels/TG_COLAVA_best.pt')
                    print('Model saved')
            sim = outputs['logits_per_text']
            for i in range(len(sim)):
                leng += 1
                res = sim[i].argmax().item()
                if res == i:
                    cn += 1
        acc = cn / leng * 100
        print('acc : ', acc)
        wandb.log({'TG_COLAVA_acc': acc})
        #torch.save(model.state_dict(), 'cocomodels/TG_COCO_epoch' + str(epoch + 1) + '.pt')
        #var = torch.var(torch.tensor(losses))
        #wandb.log({'TG_LAVA_valid_var': var})
    return


if __name__ == '__main__':
    with torch.autograd.set_detect_anomaly(True):
        main()