import re
import torch 
from torch import nn 
from transformers import CLIPProcessor, CLIPVisionModelWithProjection, CLIPTextModel 

def modify_syntax_tree(tree_text):
    modi = [] 
    anchor = 0 
    #1. First, delete '(' 
    tree_text = re.sub('\(', '', tree_text)
    
    #2. Second, isolate ever ')' 
    tree_text = re.sub('\)', ' )', tree_text)
    
    zairyo = tree_text.split()
    
    while anchor < len(zairyo): 
        target = zairyo[anchor]
        # if the target is not all capital
        if target.upper() != target: 
            modi.append(target)
            anchor += 1
        else: 
            if target in ['S', 'NP', 'VP', 'PP', 'CONJP', 'UCP', 'WHNP', 'WHADJP', 'WHPP', 'WHADVP']:
                modi.append(target)
                anchor += 1 
            elif target == ')':
                modi.append(target)
                modi.append(target)
                anchor += 1 
            else: 
                #select the ')' closest rightmost to the target
                selected = '' 
                for i in range(anchor, len(zairyo)):
                    if zairyo[i] == ')': 
                        selected = i
                        break 
                # delete the target and the selected ')'
                if selected == '':
                    return None 
                del zairyo[selected]
                del zairyo[anchor]  
                
    modi = ' '.join(modi)
                
    return modi 



def MASK(modi: str) -> torch.Tensor :
    modi_list = modi.split()
    type_list = [] 
    S = [] 
    seq_len = len(modi_list) 
    A = torch.zeros(seq_len, seq_len)
    
    for i in range(len(modi_list)): 
        if modi_list[i] in ['S', 'NP', 'VP', 'PP', 'CONJP', 'UCP', 'WHNP', 'WHADJP', 'WHPP', 'WHADVP']: 
            type_list.append([modi_list[i], 'ONT'])
        elif modi_list[i] == ')':
            if type_list[i-1][1] == 'CNT1': 
                type_list.append([modi_list[i], 'CNT2'])
            else: 
                type_list.append([modi_list[i], 'CNT1'])
        else: 
            type_list.append([modi_list[i], 'T']) 
    
    # MASK Generation Start 
    for i in range(seq_len): 
        if type_list[i][1] == "CNT1": 
            j = i 
            while type_list[j][1] != 'ONT':
                A[i][j] = 1 
                j = S.pop() 
            
            A[i][j] = 1 
            S.append(i) 
        else: 
            if type_list[i][1] != 'CNT2': 
                S.append(i) 
            for j in S: 
                A[i][j] = 1 
    
    return A 



class FFN(nn.Module):
    """
    Actually without the FFN layer, there is no non-linearity involved. That is may be why the model cannot fit
    the training queries so well
    """
    def __init__(self, hidden_size, dropout):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)


        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout) 

    def forward(self, hidden_states):
        return self.linear2(self.dropout(self.activation(self.linear1(self.dropout(hidden_states)))))



class EncoderBlock(nn.Module):
    def __init__(self, dim, head_num = 8, dropout = 0.1): 
        super().__init__() 
        self.MHA = nn.MultiheadAttention(
            embed_dim = dim, 
            num_heads = head_num, 
            dropout = dropout
            )
        self.layer_norm1 = nn.LayerNorm(dim)
        self.layer_norm2 = nn.LayerNorm(dim) 
        self.FF = FFN(dim, dropout) 
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)   
        
    def forward(self, x, mask): 
        Q = K = V = x 
        x, attn = self.MHA(Q, K, V, attn_mask = mask) 
        x = self.dropout_1(x) 
        x = x + Q 
        x = self.layer_norm1(x) 
        _x = x 
        x = self.FF(x) 
        x = self.dropout_2(x) 
        x = x + _x 
        x = self.layer_norm2(x)
        
        return x 
    
class Grammar_Encoder(nn.Module): 
    def __init__(self, dim = 512, head_num = 8, layer_nun = 6, dropout = 0.1): 
        super().__init__() 
        self.EncoderBlocks = nn.ModuleList([EncoderBlock(dim, head_num, dropout) for _ in range(layer_nun)])
        
    def forward(self, x, mask):
        for EncoderBlock in self.EncoderBlocks: 
            x = EncoderBlock(x, mask)
            
        # x : [seq_len, dim] 
        x = x.unsqueeze(0) # x : [1, seq_len, dim] 
        x = torch.mean(x, dim=1, keepdim=True) # x : [1, 1, dim]
        
        return x 

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor: 
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device = logits.device)) 

def cliploss(sim: torch.Tensor) -> torch.Tensor: 
    graph_loss = contrastive_loss(sim) 
    image_loss = contrastive_loss(sim.t()) 
    return (graph_loss + image_loss) / 2.0  


    
    
class TG_CLIP(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.vision_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.grammar_encoder = Grammar_Encoder().to(device)
        self.dimension = 512
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))
        
    def forward(self, 
                tree, 
                pixel_values, 
                return_loss = True, 
                output_attentions = None, 
                output_hidden_states = None): 
        
        output_attentions = None 
        output_hidden_states = None
        return_dict = None
        
        # encode the image
        vision_outputs = self.vision_model(
            pixel_values = pixel_values, 
            output_attentions = output_attentions, 
            output_hidden_states = output_hidden_states,
            return_dict = return_dict
            )
        vision_embeds = vision_outputs.image_embeds
        
        # encode the grammar 
        grammar_embeds = [] 
        modi_tree = tree 
        mask = [MASK(tr).to(self.device) for tr in modi_tree] 
        
        discard_index = [] 
        for i in range(len(modi_tree)): 
            inputs_gr = self.processor(
                text = modi_tree[i], 
                return_tensors = 'pt', 
                padding = True
                ).to(self.device)

            outputs_gr = self.text_model(**inputs_gr).last_hidden_state[0] 
            outputs_gr = outputs_gr[1:len(outputs_gr)-1] 
            
            try: 
                outputs_tr = self.grammar_encoder(outputs_gr, mask[i])
            except RuntimeError: 
                discard_index.append(i) 
                continue
            outputs_tr = outputs_tr.squeeze(1) 
            grammar_embeds.append(outputs_tr)
            
            
        selec_vision_embeds = [vision_embeds[i] for i in range(len(vision_embeds)) if i not in discard_index]
        vision_embeds = torch.stack(selec_vision_embeds, dim = 0) 
        
        selec_grammar_embeds = torch.stack(grammar_embeds, dim=0) 
        grammar_embeds = selec_grammar_embeds.squeeze(1) 
        
        # Normalise 
        vision_embeds = vision_embeds / vision_embeds.norm(p=2, dim=-1, keepdim=True)
        grammar_embeds = grammar_embeds / grammar_embeds.norm(p=2, dim=-1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(grammar_embeds, vision_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.t()
        
        
        loss = None 
        if return_loss: 
            loss = cliploss(logits_per_text) 
            
            
        res = {
            'loss': loss,
            'logits_per_text': logits_per_text,
            'logits_per_image': logits_per_image,
            'text_embeds': grammar_embeds,
            'image_embeds': vision_embeds
        }
        
        
        
        
        return res 