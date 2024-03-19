import torch 
from transformers import CLIPProcessor, AdamW, get_scheduler
import wandb 
import json 
from torch.utils.data import DataLoader 
from PIL import Image 
from tqdm.auto import tqdm 
from models import SemGCNCLIP, graphise_MK2 



def main():
    wandb.init()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    
    model = SemGCNCLIP(device).to(device) 
    tokenizer = model.tokenizer
    wordencoder = model.wordencoder.to(device)
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32") 
    
    imagepath = "../../../data/LAVA/images/" 
    trainpath = "../../../data/LAVA/train.json"
    evalpath = "../../../data/LAVA/val.json"
    
    
    train = json.load(open(trainpath, 'r'))
    val = json.load(open(evalpath, 'r'))
    
    # Hyperparams
    train_batch_size = 8
    eval_batch_size = 1 
    epochs = 500
    shuffle = True
    num_workers = 0
    learning_rate = 1e-5
    num_training_steps = epochs * len(train) // train_batch_size
    
    # Optim. and Scheduler
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
            train, 
            batch_size=train_batch_size, 
            shuffle=shuffle, 
            num_workers = num_workers
            )
        
        for batch in tloader: 
            text = [] 
            graph = [] 
            image = []
            for x in batch: 
                for y in x: 
                    text += y['text'] 
                    pyg = [graphise_MK2(sem, wordencoder, tokenizer, device) for sem in y['sem']] 
                    graph += pyg 
                    imgprocessed = [Image.open(imagepath + y['vision'][z] + '.png') for z in range(len(y['vision']))]
                    image += imgprocessed
                    
            pix_val = proc(images=image, return_tensors="pt", padding=True)['pixel_values'].to(device)
            
            outputs = model(
                text = text,
                graphs = graph, 
                pixel_values = pix_val,
                return_loss = True, 
                edge_weight=None, 
            )
            loss = outputs['loss'] 
            print('loss: ', loss)
            wandb.log({'MK2_graphconv_train_loss': loss})
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
            batch_size=eval_batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers
            )
        for batch in vloader: 
            text = [] 
            graph = [] 
            image = [] 
            for x in batch:
                for y in x:
                    text += y['text']
                    pyg = [graphise_MK2(sem, wordencoder, tokenizer, device) for sem in y['sem']]
                    graph += pyg
                    imgprocessed = [Image.open(imagepath + y['vision'][z] + '.png') for z in range(len(y['vision']))]
                    image += imgprocessed
                    
            pix_val = proc(images=image, return_tensors="pt", padding=True)['pixel_values'].to(device)
            
            with torch.no_grad(): 
                outputs = model(
                    text = text,
                    graphs = graph, 
                    pixel_values = pix_val, 
                    return_loss = True, 
                    edge_weight=None, 
                )
                validloss = outputs['loss'] 
                losses.append(validloss) 
                print('validloss: ', validloss)
                wandb.log({'MK2_graphconv_valid_loss': validloss})
                
                if validloss < best_valid_loss: 
                    best_valid_loss = validloss 
                    torch.save(model.state_dict(), "MK2_lavamodels/MK2_graphconv_best.pt")
                    print('saved')
    
            sim = outputs['logits_per_graph'] 
            
            for i in range(len(sim)): 
                leng += 1 
                res = sim[i].argmax().item()
                if res == i:  
                    cn += 1
                    
        acc = cn / leng * 100
        print('acc: ', acc)
        wandb.log({'MK2_graphconv_acc': acc})
        
        #torch.save(model.state_dict(), "MK2_lavamodels/K_graphconv_epoch" + str(epoch + 1) + ".pt")
        var = torch.var(torch.tensor(losses))
        wandb.log({'MK2_graphconv_valid_var': var})
        
        
    return 












if __name__ == '__main__':
    main()