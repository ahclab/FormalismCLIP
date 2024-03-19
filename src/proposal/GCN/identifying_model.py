import torch 
from torch_geometric.data import Data 
import re 
from torch import nn 
from transformers import CLIPProcessor, CLIPVisionModelWithProjection
from PIL import Image 
import requests 
from torch.nn import Linear, Parameter 
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, to_dense_adj
import math 

from transformers import CLIPTokenizer, CLIPTextModelWithProjection, CLIPTextModel   

import torch.nn.functional as F 
from torch_geometric.nn import GraphConv, global_add_pool  

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor: 
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device = logits.device)) 

def cliploss(sim: torch.Tensor) -> torch.Tensor: 
    graph_loss = contrastive_loss(sim) 
    image_loss = contrastive_loss(sim.t()) 
    return (graph_loss + image_loss) / 2.0  



class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class SelfAttention(nn.Module):

    def __init__(self, hidden_size=512):
        super(SelfAttention, self).__init__()

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.hidden_size = hidden_size

    def forward(self, particles):

        # [batch_size, num_particles, embedding_size]
        #K = self.query(particles)
        K = self.key(particles) 
        #V = self.query(particles)
        V = self.value(particles)
        Q = self.query(particles)
        
        

        # [batch_size, num_particles, num_particles]
        attention_scores = torch.matmul(Q, K.permute(0,2,1))
        attention_scores = attention_scores / math.sqrt(self.hidden_size)

        # [batch_size, num_particles, num_particles]
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # attention_probs = self.dropout(attention_probs)

        # [batch_size, num_particles, embedding_size]
        attention_output = torch.matmul(attention_probs, V)

        return attention_output

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
        self.dropout = dropout

    def forward(self, particles):
        return self.linear2(self.dropout(self.activation(self.linear1(self.dropout(particles)))))

class Intersection(nn.Module):

    def __init__(self, embedding_size, num_particles, dropout):
        super(Intersection, self).__init__()

        self.intersection_layer = SelfAttention(embedding_size)

        self.dropout = dropout

        self.num_particles = num_particles

        self.layer_norm_1 = LayerNorm(embedding_size)
        self.layer_norm_2 = LayerNorm(embedding_size)

        self.layer_norm_3 = LayerNorm(embedding_size)
        self.layer_norm_4 = LayerNorm(embedding_size)

        self.ffn = FFN(embedding_size, dropout)


    def forward(self, particles_sets):
        """
        :param particles_sets: [batch_size, num_sets, num_particles, embedding_size]
        :param weights_sets: [batch_size, num_sets, num_particles]
        :return: [batch_size, num_particles, embedding_size]
        """

        batch_size, num_sets, num_particles, embedding_size = particles_sets.shape


        # [batch_size, num_sets * num_particles, embedding_size]
        flatten_particles = particles_sets.view(batch_size, -1, embedding_size)

        # [batch_size, num_sets * num_particles, embedding_size]
        flatten_particles = self.intersection_layer(self.dropout(flatten_particles))
        flatten_particles = self.layer_norm_1(flatten_particles)

        flatten_particles = self.ffn(flatten_particles) + flatten_particles
        flatten_particles = self.layer_norm_2(flatten_particles)


        flatten_particles = self.intersection_layer(self.dropout(flatten_particles))
        flatten_particles = self.layer_norm_3(flatten_particles)

        flatten_particles = self.ffn(flatten_particles) + flatten_particles
        flatten_particles = self.layer_norm_4(flatten_particles)



        particles = flatten_particles[:, num_sets * torch.arange(self.num_particles)]

        return particles 
    
    

def graphise(sem, text_encoder, tokenizer, device): 
    # lava : lava object (comb) 
    #txt = lava['text'] 
    #sem = lava['sem']
    #people = False  
    extract_pattern = r'\(([\w-]+):'
    
    edge_index = [] 
      
    
    sem = sem.split('lambda')[-1]
    sem = sem.split('and:<t*,t>')[1:][0]  
    args = list(set(list(re.findall(extract_pattern, sem)))) 
    sem = sem.split('))') 
    sem = [elem for elem in sem if re.search('[a-zA-Z]', elem)] 
    
    #args = set(list(re.findall(extract_pattern, z)))
    rels = sem[:-1] 
    
    """
    args = argument_generator(args) 
    if args.count("person") > 1: 
        people = True 
    """
    
    # x 
    argdict = dict() 
    """
    for i in range(len(args)//2): 
        argdict[args[i*2]] = int(args[i*2+1][1]) 
    for i in range(len(list(argdict.keys()))):
        argdict[list(argdict.keys())[i]] = i 
    # Reason doing this crap : PEOPLE may make a blank in the node index
    """
    
    for i in range(len(args)): 
        argdict[args[i]] = i
        
    edgeargs = len(args) - 1 
    edgeargslist = []   
    
    argslist = list(argdict.keys()) 
    x_inputs = tokenizer(argslist, padding = True, return_tensors = "pt").to(device)
    x_outputs = text_encoder(**x_inputs)
    x = x_outputs.text_embeds 
    
    
    
    rellist = [list(re.findall(extract_pattern, line)) for line in rels] 
    
    
    for i in range(len(rellist)):
        anchor = rellist[i][0]
        for j in range(1, len(rellist[i])):
            attr = str(j-1) 
            attremb = tokenizer(attr, padding = True, return_tensors = "pt").to(device)
            attremb = text_encoder(**attremb)
            #add attreb to x 
            x = torch.cat((x, attremb.text_embeds), dim = 0) 
            edgeargs += 1
            edgeargslist.append([attr, edgeargs]) 
            
            edge_one = [argdict[anchor], edgeargs] 
            edge_one_r = reversed(edge_one) 
            edge_two = [edgeargs, argdict[rellist[i][j]]] 
            edge_two_r = reversed(edge_two)
            
            edge_index.append(edge_one)
            edge_index.append(edge_two) 
            #edge_index.append(edge_one_r)
            #edge_index.append(edge_two_r)  
            
            
    
            
    edge_index = torch.tensor(edge_index, dtype=torch.long) 
    
    #print(edge_index) 
    
    
    graph = Data(
        x = x, 
        edge_index = edge_index.t().contiguous()
    )
    
    
    return graph, argdict, edgeargslist 


def graphise_MK2(sem, text_encoder, tokenizer, device): 
    # lava : lava object (comb) 
    #txt = lava['text'] 
    #sem = lava['sem']
    #people = False  
    extract_pattern = r'\(([\w-]+):'
    
    edge_index = [] 
      
    
    sem = sem.split('lambda')[-1]
    sem = sem.split('and:<t*,t>')[1:][0]  
    args = list(set(list(re.findall(extract_pattern, sem)))) 
    sem = sem.split('))') 
    sem = [elem for elem in sem if re.search('[a-zA-Z]', elem)] 
    
    #args = set(list(re.findall(extract_pattern, z)))
    rels = sem[:-1] 
    
    """
    args = argument_generator(args) 
    if args.count("person") > 1: 
        people = True 
    """
    
    # x 
    argdict = dict() 
    """
    for i in range(len(args)//2): 
        argdict[args[i*2]] = int(args[i*2+1][1]) 
    for i in range(len(list(argdict.keys()))):
        argdict[list(argdict.keys())[i]] = i 
    # Reason doing this crap : PEOPLE may make a blank in the node index
    """
    
    for i in range(len(args)): 
        argdict[args[i]] = i
        
    edgeargs = len(args) - 1 
    edgeargslist = []   
    
    argslist = list(argdict.keys()) 
    x_inputs = tokenizer(argslist, padding = True, return_tensors = "pt").to(device)
    x_outputs = text_encoder(**x_inputs)
    x = x_outputs.text_embeds 
    
    
    
    rellist = [list(re.findall(extract_pattern, line)) for line in rels] 
    
    
    for i in range(len(rellist)):
        anchor = rellist[i][0]
        for j in range(1, len(rellist[i])):
            attr = str(j-1)
            if attr in argdict.keys():
                pass 
            else:  
                attremb = tokenizer(attr, padding = True, return_tensors = "pt").to(device)
                attremb = text_encoder(**attremb)
            #add attreb to x 
                x = torch.cat((x, attremb.text_embeds), dim = 0)
                edgeargs += 1
                argdict[attr] = edgeargs 
            
            #edgeargslist.append([attr, edgeargs]) 
            
            edge_one = [argdict[anchor], argdict[attr]] 
            edge_one_r = reversed(edge_one) 
            edge_two = [argdict[attr], argdict[rellist[i][j]]] 
            edge_two_r = reversed(edge_two)
            
            edge_index.append(edge_one)
            edge_index.append(edge_two) 
            #edge_index.append(edge_one_r)
            #edge_index.append(edge_two_r)  
            
            
    
            
    edge_index = torch.tensor(edge_index, dtype=torch.long) 
    
    #print(edge_index) 
    
    
    graph = Data(
        x = x, 
        edge_index = edge_index.t().contiguous()
    )
    
    
    return graph, argdict, edgeargslist 




class proposal(torch.nn.Module): 
    def __init__(self, dim=512):
        super(proposal, self).__init__()
        
        self.dim = dim 
        
        self.conv1 = GraphConv(512, dim) 
        self.conv2 = GraphConv(dim, dim)
        self.conv3 = GraphConv(dim, dim)
        self.conv4 = GraphConv(dim, dim)
        self.conv5 = GraphConv(dim, dim)
        
        self.lin1 = Linear(dim, dim)
        self.lin2 = Linear(dim, dim)
        
    def forward(self, x, edge_index, edge_weight=None, explain = False):
        x = self.conv1(x, edge_index, edge_weight).relu() 
        x = self.conv2(x, edge_index, edge_weight).relu()
        x = self.conv3(x, edge_index, edge_weight).relu()
        x = self.conv4(x, edge_index, edge_weight).relu()
        x = self.conv5(x, edge_index, edge_weight).relu()
        x = global_add_pool(x, batch=None) 
        x = self.lin1(x).relu() 
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x) 
        
        if explain: 
            x = F.log_softmax(x, dim=-1) 
        
        return x 
    
    

class cap_SemGCNCLIP(nn.Module):
    def __init__(self, device):
        super().__init__() 
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.wordencoder = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.textencoder = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.graph_encoder = proposal().to(device)   
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.dimension = 512
        #self.image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.logit_scale = nn.Parameter(torch.tensor(2.6592)) 
        self.concater = nn.Linear(1024, 512) 
        self.device = device 
        
    def forward(self, 
                inputs_ids,
                attention_mask, 
                graphs : list, # list of PyG graphs
                pixel_values,
                return_loss = True,
                edge_weight=None,
                output_attentions=None,
                output_hidden_states=None,):
        
        output_attentions = None 
        output_hidden_states = None
        return_dict = None  
        
        vision_outputs = self.image_encoder(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            )
        vision_embeds = vision_outputs.image_embeds 
        
        #graph_outputs = [self.graph_encoder(lavabatch[i]) for i in range(len(lavabatch))] 
        graph_outputs = [] 
        for i in range(len(graphs)): 
            graph = graphs[i][0] 
            #print(graph) 
            graph_emb = self.graph_encoder(graph.x.to(self.device), graph.edge_index.to(self.device), edge_weight).to(self.device)
            graph_outputs.append(graph_emb[0])  ### with pytorch geometric
        graph_embeds = torch.stack(graph_outputs, dim = 0) 
        
        #text_inputs = self.tokenizer(text, padding = True, return_tensors = "pt").to(self.device)
        text_outputs = self.textencoder(
            input_ids = inputs_ids,
            attention_mask = attention_mask
        ) 
        text_embeds = text_outputs.text_embeds
        
        
        # Normalisation
        vision_embeds = vision_embeds / vision_embeds.norm(p=2, dim=-1, keepdim=True)
        graph_embeds = graph_embeds / graph_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True) 
        
        concated = [] 
        
        #print('graph_embeds: ', graph_embeds.shape)
        #print('graph: ', graph.x.shape)
        #print('edge : ' , graph.edge_index.shape)
        #print('text_embeds: ', text_embeds.shape)
        
        for i in range(len(graph_embeds)): 
            concated.append(torch.cat((graph_embeds[i], text_embeds[i]), dim = 0))
        concated = torch.stack(concated, dim = 0) 
        concated = self.concater(concated)
        
        
        
        # Cos sim. 
        logit_scale = self.logit_scale.exp() 
        logits_per_graph = torch.matmul(concated, vision_embeds.t()) * logit_scale
        logits_per_image = logits_per_graph.t() 
        
        loss = None 
        if return_loss: 
            loss = cliploss(logits_per_graph) 
            
        res = {
            'loss': loss,
            'logits_per_graph': logits_per_graph,
            'logits_per_image': logits_per_image,
            'graph_embeds': graph_embeds,
            'image_embeds': vision_embeds,
            'graph_outputs': graph_outputs,
            'vision_outputs': vision_outputs,
        }
        
                                            
        
        return res     