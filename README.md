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

'''bash 
python data/LAVA/filesplit.py 

This will create the following data splits; train.json, val.json, and tst.json 
