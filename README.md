# FormalismCLIP
Codes for the paper submitted to JNLP on the 31st of Dec., 2023. 


## Data 

The research uses two datsets; MS-COCO and LAVA.　

### MS-COCO 

Follow the instruction from data/COCO/instruction.txt 


### LAVA 

Download the dataset from https://web.mit.edu/lavacorpus/ , and place all the files under the LAVA directory in this repository. 
Our research has performed two steps of modifications to the original dataset lava.json 

#### Labelling every object with its colour 

Image descriptions in lava.json don't strictly label every object with its colour, which could damage our research setups by creating ambiguities beyond 
the structural information. This preprocess was done before Chat-GPT was invented, so we handlabelled every sample, but nowadays, utilising multi-modal LLM could alleviate this burden. 


#### Missing structural information supply and pairing 

The dataset doesn't supply with structural information the sample which shares the meaning with previous samples but is matched with different visual images. Also, the experimental setups make it crucial that samples with the same text but different structures, which isn't reflected in the original dataset. For this, we implement the following code in the terminal:

```bash 
python data/LAVA/filesplit.py 
```
This will create the following data splits; train.json, val.json, and tst.json 


## Models 

Models treated throughout our research comprises of a baseline and 4 proposals. 

### Baseline 

Original CLIP trained on pure texts. 

```bash
python src/baseline/pureclip/purecocotrain.py     # Train on COCO

python src/baseline/pureclip/lava_train.py        # Train on LAVA, This could also be used for training on both COCO and LAVA, by uploading the model inside the folder 'cocomodels' before the training session 

python src/baseline/pureclip/cocotest.py          # Test on COCO, could also be used for COCO-LAVA testing 

python src/baseline/pureclip/lavatest.py          # Test on LAVA 
```


### Tree CLIP (Proposal No.1) 

CLIP finetuned with linearised syntax trees.

```bash
python src/baseline/treeclip/mirucocotrain.py     # Train on COCO

python src/baseline/treeclip/lavatrain.py        # Train on LAVA, This could also be used for training on both COCO and LAVA, by uploading the model inside the folder 'cocomodels' before the training session 

python src/baseline/treeclip/cocotest.py          # Test on COCO, could also be used for COCO-LAVA testing 

python src/baseline/treeclip/lavatest.py          # Test on LAVA 
```

### GCN CLIP (Proposal No.2) 

CLIP with semantic graphs as inputs. 

```bash


python src/proposal/GCN/cocotrain.py     # Train on COCO

python src/proposal/GCN/lavatrain.py        # Train on LAVA, This could also be used for training on both COCO and LAVA, by uploading the model inside the folder 'cocomodels' before the training session 

python src/proposal/GCN/cocotesting.py          # Test on COCO, could also be used for COCO-LAVA testing 

python src/proposal/GCN/lavatesting.py          # Test on LAVA 
```


### Separate Encodings of Syntax Tree (Proposal No.3) 

This model leveraged as language encoder the syntax tree encoder from the paper [Transformer-Based Neural Text Generation with Syntactic Guidance](https://arxiv.org/abs/2010.01737).    

In order to implement this model, follow these instructions step by step: 

1. Clone the repository https://github.com/Yinghao-Li/GuiGen inside the GuiG folder in this repository. That makes the structure 

```
GuiG/
  | 
  |------ GuiGen
  |______ CLIP
```
2. Move the CLIP folder inside the GuiGen folder.
3. Enter the CLIP folder and implement the following codes

```bash


python cocotrain.py     # Train on COCO

python lavatrain.py        # Train on LAVA, This could also be used for training on both COCO and LAVA, by uploading the model inside the folder 'cocomodels' before the training session 

python cocotesting.py          # Test on COCO, could also be used for COCO-LAVA testing 

python lavatesting.py          # Test on LAVA 
```


### Recursive Encodings of Syntax Tree (Proposal No.4) 

This model is inspired from the paper [Transformer Grammars: Augmenting Transformer Language Models with Syntactic Inductive Biases at Scale](https://aclanthology.org/2022.tacl-1.81/)

Following is the implementation codes 


```bash


python src/proposal/TG/TG_COCOtrain.py     # Train on COCO

python src/proposal/TG/TG_LAVAtrain.py        # Train on LAVA, This could also be used for training on both COCO and LAVA, by uploading the model inside the folder 'cocomodels' before the training session 

python src/proposal/TG/cocotesting.py          # Test on COCO, could also be used for COCO-LAVA testing 

python src/proposal/TG/lavatesting.py          # Test on LAVA 
```






































