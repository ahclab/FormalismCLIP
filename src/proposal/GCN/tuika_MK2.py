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
    
    # Datapaths 
    coco_image = '../../../data/COCO/'
    coco_lang  = '../../../data/COCO/dataset_coco.json'
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
    
    lava_t = json.load(open(lava_lang_train, 'r'))
    lava_v = json.load(open(lava_lang_eval, 'r'))
    lava_train = [] 
    lava_valid = [] 
    for i in lava_t: 
        lava_train.append(i[0][0]) 
        lava_train.append(i[0][1])
    for i in lava_v:
        lava_valid.append(i[0][0])
        lava_valid.append(i[0][1])
    
    # Upsampling 
    ltn = lava_train * (len(coco_train) // len(lava_train)) 
    ctn = coco_train[:len(ltn)] 
    lvn = lava_valid * (len(coco_val) // len(lava_valid)) 
    cvn = coco_val[:len(lvn)] 
    
    # Combination
    train_len = len(ltn) 
    valid_len = len(lvn) 
    train_data = [] 
    valid_data = [] 
    
    for i in range(train_len):
        singledict = {} 
        singledict['coco_image'] = ctn[i]['path']
        singledict['coco_text']  = ctn[i]['sentence']
        singledict['lava_text']  = ltn[i]['text'] 
        singledict['lava_tree']  = ltn[i]['constree'] 
        singledict['lava_image'] = ltn[i]['vision'] 
        singledict['lava_graph'] = ltn[i]['sem'] 
        
        train_data.append(singledict)
        
    for i in range(valid_len):
        singledict = {} 
        singledict['coco_image'] = cvn[i]['path']
        singledict['coco_text']  = cvn[i]['sentence']
        singledict['lava_text']  = lvn[i]['text'] 
        singledict['lava_tree']  = lvn[i]['constree'] 
        singledict['lava_image'] = lvn[i]['vision'] 
        singledict['lava_graph'] = lvn[i]['sem'] 
        
        valid_data.append(singledict) 
    
    train_batch_size = 32
    eval_batch_size = 32
    #lava_train_batch_size = 32
    #lava_eval_batch_size = 32
    epochs = 500
    shuffle = True
    num_workers = 0
    learning_rate = 1e-5
    num_training_steps = epochs * len(train_data) // train_batch_size
    num_evaluation_steps = epochs * len(valid_data) // eval_batch_size
    
    # Optimiser and Scheduler
    optim = AdamW(model.parameters(), lr=learning_rate) 
    scheduler = get_scheduler(
        "linear",
        optim,
        num_warmup_steps=0,
        num_training_steps= num_training_steps 
    )
    
    best_coco_valid_loss = 50000000
    best_lava_train_loss = 50000000
    best_coco_train_loss = 50000000
    
    progress_bar = tqdm(range(num_training_steps)) 
    
    # Freeze the vision model
    for param in model.vision_model.parameters():
        param.requires_grad = False
    
    for epoch in range(epochs): 
        coco_leng = 0 
        lava_leng = 0 
        coco_cnt  = 0 
        lava_cnt  = 0 
        
        # Training 
        model.train()
        tloader = DataLoader(
            train_data,
            batch_size = train_batch_size,
            shuffle = shuffle,
            num_workers = num_workers   
        )
        
        
        for batch in tloader: 
            # COCO Vector 
            coco_text = batch['coco_text']
            coco_graph = [] 
            mondaiji = [] 
            coco_pix = [Image.open(os.path.join(coco_image, i)) for i in batch['coco_image']]
            for i in range(len(coco_text)): 
                t = coco_text[i] 
                hanout = han(t) 
                graphised = cocograph(hanout, wordencoder, tokenizer, device)
                if graphised[0] == None: 
                    mondaiji.append(i) 
                    continue 
                coco_graph.append(graphised) 
                
            for i in mondaiji:
                coco_text.pop(i) 
                coco_pix.pop(i) 
                
            coco_pixel = proc(
                images = coco_pix, 
                return_tensors="pt",
                padding=True
            )['pixel_values'].to(device)
            
            # LAVA Vector 
            lava_text = batch['lava_text']
            lava_graph = [graphise(sem, wordencoder, tokenizer, device) for sem in batch['lava_graph']]
            lava_pix = [Image.open(lava_image + i + '.png') for i in batch['lava_image']] 
            
            lava_pixel = proc(
                images = lava_pix, 
                return_tensors="pt",
                padding=True
            )['pixel_values'].to(device)
            
            # Combination 
            text = coco_text + lava_text
            graph = coco_graph + lava_graph
            pix_val = torch.cat((coco_pixel, lava_pixel), 0)
            # ??? 
            
             
             
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            '''
            outputs = model(
                text = text, 
                graphs = graph, 
                pixel_values = pix_val,
                return_loss = True,
                edge_weight = None 
            )
            
            loss = outputs['loss'] 
            print("Loss : ", loss) 
            wandb.log({'Tuika_MK2_train_loss': loss}) 
            loss.backward()
            optim.step()
            scheduler.step()
            optim.zero_grad() 
            progress_bar.update(1) 
            
            if loss < best_coco_train_loss:
                best_coco_train_loss = loss 
                #torch.save(model.state_dict(), 'best_coco_train_model.pth')
            ''' 
                
            pass 
        
    
    
    return 










if __name__ == '__main__':
    main()