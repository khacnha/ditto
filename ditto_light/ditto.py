#   train_ditto.py
#   |
#   |-- train()
#   |   |-- loop epochs
#   |   |   |-- train_step( daet )
#   |   |   |    |-- loop batches:
#   |   |   |    |   |
#   |   |   |    |   |      forward()      xEntropy()        backward()
#   |   |   |    |   |   x ----------> y^ -----------> loss -----------> gradient
#   |   |   |    |   |
#   |   |   |-- evaluate( vandaet );  evaluate( testdaet )
#   |   |   |    |-- loop batches:
#   |   |   |    |   |                                                         y--+
#   |   |   |    |   |      forward()            smax()           p > th          |
#   |   |   |    |   |   x ----------> logits ---------> probs ---------> 0/1 ------> f1
#   |   |   |    |   |       model()                           concat()
#   |   |   |-- new best?  mark best_f1, save checkpoint
#   <---------------+
#
# show weights:  (pdb) model.bert.transformer.layer[0].attention.q_lin.weight

import os, pdb, sys, torch, random, numpy as np, argparse
import torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import sklearn.metrics as metrics

from .dataset import DittoDataset  #sn was DittoDataset
from torch.utils import data
from transformers import AutoModel, AdamW, get_linear_schedule_with_warmup
from tensorboardX import SummaryWriter
from apex import amp

lm_mp = {'roberta': 'roberta-base', 'phi3':'microsoft/Phi-3-mini-4k-instruct',   #sn added Phi-3
         'distilbert': 'distilbert-base-uncased'}

# tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct", trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct", trust_remote_code=True)

# A baseline model for EM.
class DittoModel(nn.Module):

    def __init__(self, device='cuda', lm='roberta', alpha_aug=0.8):
        super().__init__()
        if lm in lm_mp:  self.bert = AutoModel.from_pretrained(lm_mp[lm])
        else:            self.bert = AutoModel.from_pretrained(lm)
        self.device = device
        self.alpha_aug = alpha_aug
        hidden_size = self.bert.config.hidden_size  # 768
        self.dropout = nn.Dropout(0.2)
#        self.fc1    = torch.nn.Linear( hidden_size, 20 )  #sn new
#        self.fc     = torch.nn.Linear( 20, 2 )               #sn was:  self.fc  = torch.nn.Linear( hidden_size, 2 )

        encoder_layer = nn.TransformerEncoderLayer( d_model=hidden_size, nhead=4 )
        self.xf = nn.TransformerEncoder( encoder_layer, num_layers=1 ) #        self.xf  = nn.TransformerEncoder( d_model = hidden_size, nhead=4, num_encoder_layers=1 )
        self.fc  = torch.nn.Linear( hidden_size, 2 )

    #l    must convert to int16, or the default float will error "Expected tensor 
    #     for argument #1 'indices' to have one of the following scalar types: Long, Int"
    #     ditto.py(53) forward() -> enc = self.bert(x1)[0][:, 0, :]
    #     bug?  bert(x1)[0][:, 0, :] seems to encode the first token only.
    #           File "/python3.7/site-packages/torch/nn/functional.py", line 4948:   tgt_len, bsz, embed_dim = query.shape
    #           ValueError: not enough values to unpack (expected 3, got 2)
    #           query.shape is 
    """
    -> xx   = torch.nn.functional.relu( self.dropout( self.xf(enc) ) )
    (Pdb) enc.shape
    torch.Size([40, 768])  should be 40,261,768.  261 = sequence length.
    """
    def forward( self, x1, x2=None ):
        """Encode the left, right, and the concatenation of left+right.
        Args:     x1 (LongTensor): a batch of ID's
                  x2 (LongTensor, optional): a batch of ID's (augmented)
        Returns:  Tensor: binary prediction  """
        x1 = x1.to(self.device)     # (batch_size, seq_len)
        if x2 is not None:          # MixDA
            x2 = x2.to(self.device) # (batch_size, seq_len)
            enc = self.bert(torch.cat((x1, x2)))[0][:, 0, :]
            batch_size = len(x1)
            enc1 = enc[:batch_size] # (batch_size, emb_size)
            enc2 = enc[batch_size:] # (batch_size, emb_size)
            aug_lam = np.random.beta(self.alpha_aug, self.alpha_aug)
            enc = enc1 * aug_lam + enc2 * (1.0 - aug_lam)
        else:
# (Pdb) xx1 = torch.narrow( x1, 0,0,20 )
# (Pdb) yy = self.bert(xx1)[0][:,:,:]
            enc = self.bert(x1)[0][:, 0, :] #sn classification uses only the first vector, hence [:, 0, :] 
        # xx = torch.nn.functional.relu( self.dropout( self.fc1(enc) ) )
        pdb.set_trace()
        xx   = torch.nn.functional.relu( self.dropout( self.xf(enc) ) )
        out  = self.fc( xx )
        pdb.set_trace()
        return out       #sn was:    return self.fc(enc) # .squeeze() # .sigmoid()

# intent: calculate f1 over a dataset
# input : iterator  = the van or test dataset
#         threshold = the probability mark at which we accept a label.
#            for example, for label 0, if threshold = .6 and prob = .55, 
#            then we call the prediction "1"
# output; best_f1 for all batches in the dataset, and the best threshold among 0/.5/1
#a dim=1 is horizontal softmax.  [:,1] means take the 2nd column -- coordinates ( __ ,1)
#b list1 += list2 means concatenate the lists.

def best_f1_threshold( all_probs, all_y ):              # try a bunch of thresholds & take the f1-best one of the bunch
  f1 = 0.0
  for th in np.arange(0, 1, .05):
    yhat      = [1 if p > th else 0 for p in all_probs]
    new_f1    = metrics.f1_score( all_y, yhat )
    if new_f1 > f1:
      f1      = new_f1
      best_th = th
  return f1, best_th, yhat

def evaluate( model, dloader, threshold=None ):
    all_p = [];  all_y = [];  all_probs = [];  best_th = 0.5; f1 = 0.0
    model.eval()
    all_probs, all_y, yhat, yora = forwardSN( model, dloader, threshold, label=True )
    if threshold is     None: f1, threshold, yhat = best_f1_threshold( all_probs, all_y )  #sn functionized best_f1_threshold
    if threshold is not None: 
        yhat       = [1 if p > threshold else 0 for p in all_probs] # yhat is self-accumulating
        rates = txr( all_y, yhat )                                  # was (f1):  tp, fp, tn, fn, f1 = f1s( all_y, yhat )
    print( f'ditto.py evaluate(). \n ytru = {all_y} \n yhat = {yhat} ' )
    return rates, threshold, yhat                                   # was (f1):  return f1, threshold, yhat

""" using f1.  before tru +- rates
def evaluate( model, dloader, threshold=None ):
    all_p = [];  all_y = [];  all_probs = [];  best_th = 0.5; f1 = 0.0
    model.eval()
    all_probs, all_y = forwardSN( model, dloader, threshold, label=True )
    if threshold is     None: f1, threshold, yhat = best_f1_threshold( all_probs, all_y )  #sn functionized best_f1_threshold
    if threshold is not None: 
        yhat       = [1 if p > threshold else 0 for p in all_probs] # yhat is self-accumulating
        tp, fp, tn, fn, f1 = f1s( all_y, yhat )
    print( f'ditto.py \n  ytru = {all_y} \n yhat = {yhat} ' )
    return f1, threshold, yhat
"""

def f1s(ytru, yhat):                                        # calculate accuracies
    tp = 0; fp = 0; tn = 0; fn = 0
    for i in range(len(yhat)): 
        if ytru[i] == yhat[i] == 1            : tp += 1
        if ytru[i] == yhat[i] == 0            : tn += 1
        if yhat[i] == 0 and ytru[i] != yhat[i]: fn += 1
        if yhat[i] == 1 and ytru[i] != yhat[i]: fp += 1
    f1 = 2*tp / ( 2*tp + fp +fn )
    return tp, fp, tn, fn, f1

def txr(ytru, yhat):                                        # calculate tpr and tnr
    tp = 0; tn = 0; tpL = []; tnL = []; fpL = []; fnL = [];
    for i in range(len(ytru)):
        xx = fpL.append(i) if (ytru[i] == 0 and yhat[i] == 1) else None
        yy = fnL.append(i) if (ytru[i] == 1 and yhat[i] == 0) else None
    for i in range(len(ytru)):
        if ytru[i] == yhat[i] and yhat[i] == 1: tpL.append(1)           # failed: tpL = [ i for i in ytru if (ytru[i] == yhat[i] == 1) ]; tp   = len(tpL)
        if ytru[i] == yhat[i] and yhat[i] == 0: tnL.append(0)           # failed: tnL = [ i for i in ytru if (ytru[i] == yhat[i] == 0) ]; tn   = len(tnL)
    tp = len(tpL);  tn = len(tnL);
    pos = [ i for i in ytru if i == 1 ];  posK = len(pos);  tpr = tp / posK if posK >= 1 else -1
    neg = [ i for i in ytru if i == 0 ];  negK = len(neg);  tnr = tn / negK if negK >= 1 else -1
    rates = { 'tpr':tpr, 'tp':tp, 'posK':posK, 'tnr':tnr, 'tn':tn, 'negK':negK, 'fpL':fpL, 'fnL':fnL }
    return rates

#b threshold provided so we can calculate yhat. yhat is self-accumulating
#c label     provided so we can calculate f1.   yora means yhat or accuracy
def forwardSN( model, train_iter, threshold=None, label=True ):  #sn: new section.  calculate f1 for train set
    all_y = [];  all_probs = [];  yhat = []
    with torch.no_grad():
      for i, batch in enumerate( train_iter ):
        x, y       = batch
        all_y     += y.tolist()
        logits     = model(x)
        probs      = logits.softmax(dim=1)[:, 1]        #a        probs      = model(x).softmax( dim=1 )[:,1]
        all_probs += probs.tolist()
    if threshold == None:  return all_probs, all_y  # return to evaluate().  cannot calculate yhat and f1
    yhat = [1 if p > threshold else 0 for p in all_probs] #b
    yora = txr( all_y, yhat ) if label == True else yhat  #c  # was (f1):   yorf = f1s( all_y, yhat ) if label == True else yhat  #c
    print( f'ditto.py forward(). yora = {yora}' )                       # was (f1):   print( f'ditto.py. yorf = {yorf}' )
    return all_probs, all_y, yhat, yora

"""          using f1.  before switching to True +/- Rates
#b threshold provided so we can calculate yhat. yhat is self-accumulating
#c label     provided so we can calculate f1.   yorf means yhat or f1
def forwardSN( model, train_iter, threshold=None, label=True ):  #sn: new section.  calculate f1 for train set
    all_y = [];  all_probs = [];  yhat = []
    with torch.no_grad():
      for i, batch in enumerate( train_iter ):
        x, y       = batch
        all_y     += y.tolist()
        logits     = model(x)
        probs      = logits.softmax(dim=1)[:, 1]        #a        probs      = model(x).softmax( dim=1 )[:,1]
        all_probs += probs.tolist()
    if threshold == None:  return all_probs, all_y  # return to evaluate().  cannot calculate yhat and f1
    yhat = [1 if p > threshold else 0 for p in all_probs] #b
    yorf = f1s( all_y, yhat ) if label == True else yhat  #c
    print( f'ditto.py. yorf = {yorf}' )
    return all_probs, all_y
"""

# intent: train 1 epoch
# inputs: train_iter (Iterator): training dataset.
#         model (DMModel): the model
#         optimizer (Optimizer): the optimizer (Adam or AdamW)
#         scheduler (LRScheduler): learning rate scheduler
#         hp (Namespace): other hyper-parameters (e.g., fp16)
# output: none.  changes made to the input "model"

def train_step(train_iter, model, optimizer, scheduler, hp ):  #sn added threshold for debugging.  delete threshold when done
    criterion = nn.CrossEntropyLoss()       # criterion = nn.MSELoss()
    for i, batch in enumerate(train_iter):  # loop batch through the dataset
        model.zero_grad()                   # this was missing from Ditto!
        optimizer.zero_grad()
        if len(batch) == 2:
            x, y = batch
            prediction = model(x)
        else:
            x1, x2, y = batch
            prediction = model(x1, x2)
        loss = criterion(prediction, y.to(model.device))
        if hp.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:   loss.backward()
        optimizer.step()                    # nudge weights
        scheduler.step()
        print(f"ditto.py.  batch: {i}, loss: {round(loss.item(),5)}")  #sn  was:   if i % 10 == 0:    print(f"step: {i}, loss: {loss.item()}")  # monitoring
        del loss

# intent: loop each epoch and train
# input : 3 vatt datasets,  run_tag (str): the tag of the run
#        hp (Namespace): Hyper-parameters (e.g., batch_size,  learning rate, fp16)
# output:  None
def train(trainset, validset, testset, run_tag, hp):
    padder     = trainset.pad;   best_van_acc = 0.0; th = .5
    train_iter = data.DataLoader(dataset=trainset, batch_size=hp.batch_size   , shuffle=True , num_workers=0, collate_fn=padder)
    valid_iter = data.DataLoader(dataset=validset, batch_size=hp.batch_size*16, shuffle=False, num_workers=0, collate_fn=padder)
    test_iter  = data.DataLoader(dataset=testset , batch_size=hp.batch_size*16, shuffle=False, num_workers=0, collate_fn=padder)
    writer     = SummaryWriter( log_dir=hp.logdir )           # log with tensorboardX
    num_steps  = (len(trainset) // hp.batch_size) * hp.n_epochs
    model, optimizer, scheduler, epoch = load_model( hp, num_steps )  #sn:  the body of load_model() was here.  i removed and functionized it.

    for epoch in range( epoch+1, epoch + hp.n_epochs + 1):
        model.train()
        train_step( train_iter, model, optimizer, scheduler, hp )
        model.eval()
        acc, th, yhat = evaluate( model, valid_iter, .95 )       # was:  van_f1, th, yhat = evaluate( model, valid_iter )
        van_acc = ( acc['tpr'] + acc['tnr'] ) / 2
        if van_acc        > best_van_acc:                       # we mark the dev f1, but print best_test_f1 even though it's worse
            best_van_acc  = van_acc
            if hp.save_model:  save_checkpoint( hp, model, optimizer, scheduler, epoch )
        print( f"epoch {epoch}: van_acc={round(van_acc,4)} threshold={round(th,2)}, best_van_acc={round(best_van_acc,4)}\n" )
        writer.add_scalars( run_tag, {'f1': van_acc}, epoch )  # log
    writer.close()
"""                     using f1.  (before switching to True +- rates
def train(trainset, validset, testset, run_tag, hp):
    padder     = trainset.pad;   best_van_f1 = 0.0; th = .5
    train_iter = data.DataLoader(dataset=trainset, batch_size=hp.batch_size   , shuffle=True , num_workers=0, collate_fn=padder)
    valid_iter = data.DataLoader(dataset=validset, batch_size=hp.batch_size*16, shuffle=False, num_workers=0, collate_fn=padder)
    test_iter  = data.DataLoader(dataset=testset , batch_size=hp.batch_size*16, shuffle=False, num_workers=0, collate_fn=padder)
    writer     = SummaryWriter( log_dir=hp.logdir )  # log with tensorboardX
    num_steps  = (len(trainset) // hp.batch_size) * hp.n_epochs
    model, optimizer, scheduler, epoch = load_model( hp, num_steps )  #sn:  the body of load_model() was here.  i removed and functionized it.

    for epoch in range( epoch+1, epoch + hp.n_epochs + 1):
        model.train()
        train_step( train_iter, model, optimizer, scheduler, hp )
        model.eval()
        van_f1, th, yhat = evaluate( model, valid_iter )
        if van_f1        > best_van_f1:      # we mark the dev f1, but print best_test_f1 even though it's worse
            best_van_f1  = van_f1
            if hp.save_model:  save_checkpoint( hp, model, optimizer, scheduler, epoch )
        print( f"epoch {epoch}: van_f1={round(van_f1,4)} threshold={round(th,2)}, best_van_f1={round(best_van_f1,4)}\n" )
        writer.add_scalars( run_tag, {'f1': van_f1}, epoch )  # log
    writer.close()
"""

def save_checkpoint( hp, model, optimizer, scheduler, epoch ):
  if not os.path.exists( hp.logdir ):   os.makedirs( hp.logdir )
  ckpt_path = os.path.join(hp.logdir, hp.ckfile)
  ckpt = { 'model': model.state_dict(),     'optimizer': optimizer.state_dict(),
       'scheduler': scheduler.state_dict(), 'epoch'    : epoch }
  torch.save(ckpt, ckpt_path)

#j # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference
#  must have strict=False or else error:
#  *** RuntimeError: Error(s) in loading state_dict for DittoModel:
#  Missing key(s) in state_dict: "bert.embeddings.word_embeddings.weight", "bert.embeddings.position_embeddings.weight", "bert.embeddings.LayerNorm.weight", "bert.embeddings.LayerNorm.bias", "bert.transformer.layer.0.attention.q_lin.weight", "bert.transformer.layer.0.attention.q_lin.bias", "bert.transformer.layer.0.attention.k_lin.weight", "bert.transformer.layer.0.attention.k_lin.bias", "bert.transformer.layer.0.attention.v_lin.weight", "bert.transformer.layer.0.attention.v_lin.bias", "bert.transformer.layer.0.attention.out_lin.weight", "bert.transformer.layer.0.attention.out_lin.bias", "bert.transformer.layer.0.sa_layer_norm.weight", "bert.transformer.layer.0.sa_layer_norm.bias", "bert.transformer.layer.0.ffn.lin1.weight", "bert.transformer.layer.0.ffn.lin1.bias", "bert.transformer.layer.0.ffn.lin2.weight", "bert.transformer.layer.0.ffn.lin2.bias", "bert.transformer.layer.0.output_layer_norm.weight", "bert.transformer.layer.0.output_layer_norm.bias", "bert.transformer.layer.1.attention.q_lin.weight", "bert.transformer.layer.1.attention.q_lin.bias", "bert.transformer.layer.1.attention.k_lin.weight", "bert.transformer.layer.1.attention.k_lin.bias", "bert.transformer.layer.1.attention.v_lin.weight", "bert.transformer.layer.1.attention.v_lin.bias", "bert.transformer.layer.1.attention.out_lin.weight", "bert.transformer.layer.1.attention.out_lin.bias", "bert.transformer.layer.1.sa_layer_norm.weight", "bert.transformer.layer.1.sa_layer_norm.bias", "bert.transformer.layer.1.ffn.lin1.weight", "bert.transformer.layer.1.ffn.lin1.bias", "bert.transformer.layer.1.ffn.lin2.weight", "bert.transformer.layer.1.ffn.lin2.bias", "bert.transformer.layer.1.output_layer_norm.weight", "bert.transformer.layer.1.output_layer_norm.bias", "bert.transformer.layer.2.attention.q_lin.weight", "bert.transformer.layer.2.attention.q_lin.bias", "bert.transformer.layer.2.attention.k_lin.weight", "bert.transformer.layer.2.attention.k_lin.bias", "bert.transformer.layer.2.attention.v_lin.weight", "bert.transformer.layer.2.attention.v_lin.bias", "bert.transformer.layer.2.attention.out_lin.weight", "bert.transformer.layer.2.attention.out_lin.bias", "bert.transformer.layer.2.sa_layer_norm.weight", "bert.transformer.layer.2.sa_layer_norm.bias", "bert.transformer.layer.2.ffn.lin1.weight", "bert.transformer.layer.2.ffn.lin1.bias", "bert.transformer.layer.2.ffn.lin2.weight", "bert.transformer.layer.2.ffn.lin2.bias", "bert.transformer.layer.2.output_layer_norm.weight", "bert.transformer.layer.2.output_layer_norm.bias", "bert.transformer.layer.3.attention.q_lin.weight", "bert.transformer.layer.3.attention.q_lin.bias", "bert.transformer.layer.3.attention.k_lin.weight", "bert.transformer.layer.3.attention.k_lin.bias", "bert.transformer.layer.3.attention.v_lin.weight", "bert.transformer.layer.3.attention.v_lin.bias", "bert.transformer.layer.3.attention.out_lin.weight", "bert.transformer.layer.3.attention.out_lin.bias", "bert.transformer.layer.3.sa_layer_norm.weight", "bert.transformer.layer.3.sa_layer_norm.bias", "bert.transformer.layer.3.ffn.lin1.weight", "bert.transformer.layer.3.ffn.lin1.bias", "bert.transformer.layer.3.ffn.lin2.weight", "bert.transformer.layer.3.ffn.lin2.bias", "bert.transformer.layer.3.output_layer_norm.weight", "bert.transformer.layer.3.output_layer_norm.bias", "bert.transformer.layer.4.attention.q_lin.weight", "bert.transformer.layer.4.attention.q_lin.bias", "bert.transformer.layer.4.attention.k_lin.weight", "bert.transformer.layer.4.attention.k_lin.bias", "bert.transformer.layer.4.attention.v_lin.weight", "bert.transformer.layer.4.attention.v_lin.bias", "bert.transformer.layer.4.attention.out_lin.weight", "bert.transformer.layer.4.attention.out_lin.bias", "bert.transformer.layer.4.sa_layer_norm.weight", "bert.transformer.layer.4.sa_layer_norm.bias", "bert.transformer.layer.4.ffn.lin1.weight", "bert.transformer.layer.4.ffn.lin1.bias", "bert.transformer.layer.4.ffn.lin2.weight", "bert.transformer.layer.4.ffn.lin2.bias", "bert.transformer.layer.4.output_layer_norm.weight", "bert.transformer.layer.4.output_layer_norm.bias", "bert.transformer.layer.5.attention.q_lin.weight", "bert.transformer.layer.5.attention.q_lin.bias", "bert.transformer.layer.5.attention.k_lin.weight", "bert.transformer.layer.5.attention.k_lin.bias", "bert.transformer.layer.5.attention.v_lin.weight", "bert.transformer.layer.5.attention.v_lin.bias", "bert.transformer.layer.5.attention.out_lin.weight", "bert.transformer.layer.5.attention.out_lin.bias", "bert.transformer.layer.5.sa_layer_norm.weight", "bert.transformer.layer.5.sa_layer_norm.bias", "bert.transformer.layer.5.ffn.lin1.weight", "bert.transformer.layer.5.ffn.lin1.bias", "bert.transformer.layer.5.ffn.lin2.weight", "bert.transformer.layer.5.ffn.lin2.bias", "bert.transformer.layer.5.output_layer_norm.weight", "bert.transformer.layer.5.output_layer_norm.bias", "fc.weight", "fc.bias". 
#  Unexpected key(s) in state_dict: "model", "optimizer", "scheduler", "epoch".

def load_model( hp, num_steps ):  # num_steps used by learning rate scheduler, not needed for evaluate()
    # initialize model, optimizer, and LR scheduler
    epoch      = 0
    device     = 'cuda' if torch.cuda.is_available() else 'cpu'
    model      = DittoModel(device=device, lm=hp.lm, alpha_aug=hp.alpha_aug)
    model      = model.cuda()
    optimizer  = AdamW( model.parameters(), lr=hp.lr )

    if hp.fp16:  model, optimizer = amp.initialize( model, optimizer, opt_level='O2' )
    #sn was:   num_steps = (len(trainset) // hp.batch_size) * hp.n_epochs
    scheduler = get_linear_schedule_with_warmup( optimizer, num_warmup_steps=0, num_training_steps=num_steps )
    writer    = SummaryWriter( log_dir=hp.logdir )  # log with tensorboardX
    #h https://brsoff.github.io/tutorials/beginner/saving_loading_models.html
    #  documentation is bad?  checkpoint['model_state_dict'] --> KeyError
    ckpath = os.path.join( hp.logdir, hp.ckfile ) #'checkpoints/Structured/Beer/model.pt'
    if  ( hp.load_ckp is not None ) and ( os.path.isfile( ckpath ) ):
        checkpoint = torch.load(   ckpath )
        model.load_state_dict(     checkpoint['model'] )  #j
        optimizer.load_state_dict( checkpoint['optimizer'] )
        scheduler.load_state_dict( checkpoint['scheduler'] )  # need this line, otherwise model restart with bad f1
        epoch                    = checkpoint['epoch']
    return model, optimizer, scheduler, epoch
