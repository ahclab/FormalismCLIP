import torch 
from transformers import CLIPProcessor, AdamW, get_scheduler
import wandb 
import json 
from torch.utils.data import DataLoader 
from PIL import Image 
from tqdm.auto import tqdm 
from models import SemGCNCLIP, graphise 
import hanlp 
from torch_geometric.data import Data 
import re 
import os 




def is_digit(char):
    try:
        int(char)
        return True 
    except ValueError:
        return False

def cocograph(doc, text_encoder, tokenizer, device):
    tok = doc['tok'] 
    rel = doc['sdp/pas']
    
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



def main(): 
    wandb.init() 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    han = hanlp.load(hanlp.pretrained.mtl.UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_XLMR_BASE)
    model = SemGCNCLIP(device).to(device)
    tokenizer = model.tokenizer
    wordencoder = model.wordencoder.to(device)  
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    coco_lang = '../../../data/COCO/dataset_coco.json' 
    coco_image = '../../../data/COCO/' 
    lava_image = '../../../data/LAVA/images/' 
    lava_lang_train = '../../../data/LAVA/train.json'
    lava_lang_eval = '../../../data/LAVA/val.json'
    
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
    
    
    optim = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_scheduler(
        "linear",
        optim,
        num_warmup_steps=0,
        num_training_steps = coco_num_training_steps + lava_num_training_steps
    )
    
    best_coco_valid_loss = 50000000
    best_lava_train_loss = 50000000
    best_coco_train_loss = 50000000
    
    progress_bar = tqdm(range(coco_num_training_steps))
    
    # Freeze the vision model
    for param in model.image_encoder.parameters():
        param.requires_grad = False
        
    for epoch in range(epochs): 
        leng = 0 
        cn = 0 
        
        # Training 
        model.train() 
        coco_tloader = DataLoader(
            coco_train, 
            batch_size = coco_train_batch_size,
            shuffle = shuffle, 
            num_workers = num_workers
        )
        
        lava_tloader = DataLoader(
            lava_train, 
            batch_size=lava_train_batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
        
        for batch in coco_tloader:
            text = batch['sentence']
            graph = []
            mondaiji = []
            image = [Image.open(os.path.join(coco_image, i)) for i in batch['path']]
            for i in range(len(text)): 
                t = text[i]
                hanout = han(t)
                graphised = cocograph(hanout, wordencoder, tokenizer, device)
                if graphised[0] == None:
                    mondaiji.append(i)
                    continue
                graph.append(graphised)
                
            for i in mondaiji:
                text.pop(i)
                image.pop(i)
                
            pix_val = proc(
                images = image,
                return_tensors="pt",
                padding=True
            )['pixel_values'].to(device) 
            
            outputs = model(
                text = text,
                graphs = graph,
                pixel_values = pix_val,
                return_loss = True,
                edge_weight = None
            )
            
            loss = outputs['loss']
            print("loss: ", loss)
            wandb.log({'Tuika_graphconv_train_loss': loss})
            loss.backward()
            optim.step()
            scheduler.step()
            optim.zero_grad()
            progress_bar.update(1)
            
            if loss < best_coco_train_loss:
                best_coco_train_loss = loss
    
        
        for batch in lava_tloader:
            text = [] 
            graph = []
            image = []
            for x in batch:
                for y in x:
                    text += y['text']
                    pyg = [graphise(sem, wordencoder, tokenizer, device) for sem in y['sem']]
                    graph += pyg
                    imgprocessed = [Image.open(lava_image + y['vision'][z] + '.png') for z in range(len(y['vision']))]
                    image += imgprocessed
                    
            pix_val = proc(
                images = image,
                return_tensors = 'pt',
                padding = True
            )['pixel_values'].to(device)
            
            
            outputs = model(
                text = text,
                graphs = graph,
                pixel_values = pix_val,
                return_loss = True,
                edge_weight = None,
            )
            loss = outputs['loss']
            print('loss: ', loss)
            wandb.log({'Tuika_graphconv_train_loss': loss})
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
            graph = [] 
            mondaiji = [] 
            image = [Image.open(os.path.join(coco_image, i)) for i in batch['path']]
            for i in range(len(text)): 
                t = text[i]
                hanout = han(t)
                graphised = cocograph(hanout, wordencoder, tokenizer, device)
                if graphised[0] == None:
                    mondaiji.append(i)
                    continue
                graph.append(graphised)
            
            for i in mondaiji:
                text.pop(i)
                image.pop(i)
                
            pix_val = proc(
                images = image,
                return_tensors="pt",
                padding=True
            )['pixel_values'].to(device)
            
            
            with torch.no_grad():
                outputs = model(
                    text = text,
                    graphs = graph,
                    pixel_values = pix_val,
                    return_loss = True,
                    edge_weight = None
                )
                validloss = outputs['loss']
                losses.append(validloss)
                print('validloss: ', validloss)
                wandb.log({'Tuika_COCO_graphconv_valid_loss': validloss})
                
                if validloss < best_coco_valid_loss:
                    best_coco_valid_loss = validloss
                    torch.save(model.state_dict(), 'tuikamodels/best_tuikaGCN_model.pt')
                    print('model saved')
                    
            sim = outputs['logits_per_graph'] 
            
            for i in range(len(sim)):
                leng += 1
                res = sim[i].argmax().item()
                if res == i:
                    cn += 1
                    
        acc = cn / leng * 100
        print('COCO accuracy: ', acc)
        wandb.log({'Tuika_COCO_graphconv_accuracy': acc})
        
        var = torch.var(torch.tensor(losses))
        wandb.log({'Tuika_COCO_graphconv_variance': var})
                

    return 





if __name__ == '__main__':
    main()