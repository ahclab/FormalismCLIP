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
    
    cocopath = '../../../data/COCO/dataset_coco.json' 
    imagepath = '../../../data/COCO/' 
    
    coco = json.load(open(cocopath, 'r'))
    coco = coco['images']
    train = [i for i in coco if i['split'] == 'train']
    val = [i for i in coco if i['split'] == 'val']
    newtrain = [{'path' : os.path.join(i['filepath'], i['filename']),
             'sentence' : re.sub(r'[^\w\s]', '', i['sentences'][0]['raw'])} for i in train]
    newval = [{'path' : os.path.join(i['filepath'], i['filename']),
             'sentence' : re.sub(r'[^\w\s]', '', i['sentences'][0]['raw'])} for i in val]
    
    # Hyperparams 
    train_batch_size = 16
    eval_batch_size = 16
    epochs = 10
    shuffle = True
    num_workers = 0
    learning_rate = 1e-5
    num_training_steps = epochs * len(newtrain) // train_batch_size
    
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
            num_workers=num_workers
        )
        
        for batch in tloader: 
            text = batch['sentence'] 
            graph = []
            mondaiji = []
            image = [Image.open(os.path.join(imagepath, i)) for i in batch['path']] 
            for i in range(len(text)): 
                #print("text is :",i)
                t = text[i]
                hanout = han(t) 
                graphised= cocograph(hanout, wordencoder, tokenizer, device)
                if graphised[0] == None:
                    print("mondaiji: ", i) 
                    mondaiji.append(i)
                    continue
                graph.append(graphised)
            
            for i in mondaiji:
                text.pop(i)
                image.pop(i)
            
            pix_val = proc(
                images = image, 
                return_tensors = "pt",
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
            print("loss: ", loss) 
            wandb.log({'COCO_GCN_loss': loss})
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
            num_workers=num_workers
        )
        
        for batch in vloader: 
            text = batch['sentence'] 
            graph = [] 
            mondaiji = []
            image = [Image.open(os.path.join(imagepath, i)) for i in batch['path']] 
            for i in range(len(text)):
                t = text[i] 
                hanout = han(t) 
                graphised = cocograph(hanout, wordencoder, tokenizer, device)
                if graphised[0] == None:
                    print("mondaiji: ", i) 
                    # remove the same index from text and image 
                    mondaiji.append(i)
                    continue
                graph.append(graphised)
                
            for i in mondaiji:
                text.pop(i)
                image.pop(i)
                
            pix_val = proc(
                images = image, 
                return_tensors = "pt",
                padding = True
            )['pixel_values'].to(device) 
            
            
            with torch.no_grad():
                outputs = model(
                    text = text, 
                    graphs = graph, 
                    pixel_values = pix_val, 
                    return_loss = True, 
                    edge_weight = None,
                )
            
                validloss = outputs['loss'] 
                losses.append(validloss) 
                print("validloss: ", validloss)
                wandb.log({'COCO_GCN_validloss': validloss})
                
                if validloss < best_valid_loss: 
                    best_valid_loss = validloss
                    torch.save(model.state_dict(), 'cocomodels/COCO_GCN_best.pt')
                    print("Model saved")
                    
            sim = outputs['logits_per_graph']
            
            
            for i in range(len(sim)): 
                leng += 1
                res = sim[i].argmax().item() 
                if res == i: 
                    cn += 1 
                    
        acc = cn / leng * 100
        print("acc: ", acc)
        wandb.log({'COCO_GCN_acc': acc})
        
        torch.save(model.state_dict(), 'cocomodels/COCO_GCN_epoch' + str(epoch + 1) + '.pt')
        var = torch.var(torch.tensor(losses))
        wandb.log({'COCO_GCN_valid_var': var})
            
    
    return 





if __name__ == '__main__':
    main()