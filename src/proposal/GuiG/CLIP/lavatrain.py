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


def main(): 
    wandb.init() 
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    model = GuiGCLIP(device).to(device)
    model.load_state_dict(torch.load('cocomodels/GuiG_COCO_best.pt'))
    proc = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    
    imagepath = "../../../../../data/LAVA/images/" 
    trainpath = "../../../../../data/LAVA/train.json"
    evalpath = "../../../../../data/LAVA/val.json"
    
    
    newtrain = json.load(open(trainpath, 'r'))
    newval = json.load(open(evalpath, 'r'))
    
    dict_dir = os.path.join('../data/train', 'para_train_dict.pt') 
    w2i_dict = torch.load(dict_dir) 
    syntax_dict = w2i_dict['syntax']
    level_dict = w2i_dict['level']
    
    
    
    # Hyperparams 
    train_batch_size = 8
    eval_batch_size = 1
    epochs = 1000
    shuffle = True
    num_workers = 0
    learning_rate = 1e-5
    num_training_steps = epochs * len(newtrain) // train_batch_size
    optim = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_scheduler(
        'linear',
        optim,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    best_valid_loss = 50000
    best_train_loss = 50000
    progress_bar = tqdm(range(num_training_steps))
    
    # Freeze the vision model
    for param in model.image_encoder.parameters():
        param.requires_grad = False
    for epoch in range(epochs):
        leng = 0 
        cn = 0
        # Training
        model.train()
        tloader = DataLoader(
            newtrain, 
            batch_size=train_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        ) 
        for batch in tloader: 
            text = [] 
            tree = []
            image = [] 
            for x in batch: 
                for y in x: 
                    text += y['text'] 
                    tree += y['constree'] 
                    imgprocessed = [
                        Image.open(imagepath + y['vision'][z] + '.png')
                        for z in range(len(y['vision']))
                    ]
                    image += imgprocessed 
            
            """
            image = [
                Image.open(os.path.join(imagepath, i)) 
                for i in batch['path']
            ]
            """ 
            
            
            #text = batch['sentence'] 
            
            #tree = batch['tree'] 
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
            loss = outputs['loss'] 
            wandb.log({'GuiG_COLAVA_loss': loss}) 
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
            newval, 
            batch_size=eval_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
        for batch in vloader: 
            text = [] 
            tree = []
            image = []
            for x in batch:
                for y in x:
                    text += y['text']
                    tree += y['constree']
                    imgprocessed = [
                        Image.open(imagepath + y['vision'][z] + '.png')
                        for z in range(len(y['vision']))
                    ]
                    image += imgprocessed
            """        
            image = [
                Image.open(os.path.join(imagepath, i)) 
                for i in batch['path']
            ]
            """ 
            
            #text = batch['sentence'] 
            
            #tree = batch['tree'] 
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
    
            with torch.no_grad(): 
                outputs = model(
                    texts = text, 
                    syn_seqs = syn_seqs,
                    lvl_seqs = lvl_seqs,
                    pos_seqs = pos_seqs,
                    path_masks = path_masks,
                    pixel_values = pix_val,
                    return_loss = True
                )
                validloss = outputs['loss']
                losses.append(validloss) 
                wandb.log({'GuiG_COLAVA_validloss': validloss}) 
                if validloss < best_valid_loss: 
                    best_valid_loss = validloss
                    torch.save(model.state_dict(), 'colavamodels/GuiG_COLAVA_best.pt')
                    print('best model saved')
            
            sim = outputs['logits_per_text'] 
            for i in range(len(sim)): 
                leng += 1 
                res = sim[i].argmax().item() 
                if res == i: 
                    cn += 1 
        acc = cn / leng * 100 
        print('Accuracy: ', acc) 
        wandb.log({'GuiG_COLAVA_acc': acc})
        #var = torch.var(torch.tensor(losses))
        #wandb.log({'GuiG_LAVA_var': var}) 
                
    return 




if __name__ == '__main__':
    main()