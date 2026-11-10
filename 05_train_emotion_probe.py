#!/usr/bin/env python3
"""Train a frozen-representation linear emotion probe for Sarvam-1."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

def prepare_dataframe(path, text_column="hinglish_genz_text", label_column="label"):
    df=pd.read_csv(path)[[text_column,label_column]].rename(columns={text_column:"text",label_column:"label"})
    df.text=df.text.astype("string").str.strip(); df.label=df.label.astype("string").str.strip()
    df=df.dropna().loc[(df.text!="")&(df.label!="")].drop_duplicates().reset_index(drop=True)
    labels=sorted(df.label.unique().tolist()); mapping={x:i for i,x in enumerate(labels)}
    df["label_id"]=df.label.map(mapping).astype(int)
    return df, labels

def stratified_split(labels, seed=42):
    rng=np.random.default_rng(seed); groups={int(y):np.where(labels==y)[0].tolist() for y in np.unique(labels)}
    train=[]; validation=[]; test=[]
    for indices in groups.values():
        rng.shuffle(indices); n=len(indices); n_test=max(1,round(n*.15)); n_val=max(1,round(n*.15))
        test += indices[:n_test]; validation += indices[n_test:n_test+n_val]; train += indices[n_test+n_val:]
    return np.array(train),np.array(validation),np.array(test)

def valid_mask(input_ids, attention_mask, tokenizer):
    mask=attention_mask.bool().clone(); special=set(tokenizer.all_special_ids)
    for token_id in special: mask &= input_ids.ne(token_id)
    return mask

def extract_representations(model, tokenizer, texts, batch_size=4, max_length=256):
    model.eval(); vectors=[]; device=next(model.parameters()).device
    with torch.inference_mode():
        for start in tqdm(range(0,len(texts),batch_size),desc="Extracting frozen representations"):
            enc=tokenizer(texts[start:start+batch_size],return_tensors="pt",padding=True,truncation=True,max_length=max_length)
            if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
            enc={k:v.to(device) for k,v in enc.items()}
            mask=valid_mask(enc["input_ids"],enc["attention_mask"],tokenizer)
            # The probe needs decoder hidden states, not vocabulary logits.
            out=model.model(**enc,output_hidden_states=False,use_cache=False)
            hidden=out.last_hidden_state
            weights=mask.to(hidden.device).unsqueeze(-1).to(hidden.dtype)
            pooled=(hidden*weights).sum(1)/weights.sum(1).clamp_min(1)
            vectors.append(pooled.float().cpu())
    return torch.cat(vectors)

def classification_metrics(y_true,y_pred,num_classes):
    cm=np.zeros((num_classes,num_classes),dtype=int)
    for truth,pred in zip(y_true,y_pred): cm[int(truth),int(pred)]+=1
    precision=[]; recall=[]; f1=[]
    for c in range(num_classes):
        tp=cm[c,c]; fp=cm[:,c].sum()-tp; fn=cm[c,:].sum()-tp
        p=tp/(tp+fp) if tp+fp else 0.; r=tp/(tp+fn) if tp+fn else 0.; precision.append(p); recall.append(r); f1.append(2*p*r/(p+r) if p+r else 0.)
    accuracy=float(np.trace(cm)/max(1,cm.sum())); return {"accuracy":accuracy,"macro_f1":float(np.mean(f1)),"micro_f1":accuracy,"precision":precision,"recall":recall,"f1":f1,"confusion_matrix":cm.tolist()}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model-id',default='meta-llama/Llama-3.1-8B-Instruct'); p.add_argument('--csv-path',default='data/mlt_hinghlish_dataset_clean.csv'); p.add_argument('--output-dir',default='results/llama31_8b_neuron_analysis/04_probe'); p.add_argument('--batch-size',type=int,default=4); p.add_argument('--max-length',type=int,default=256); p.add_argument('--epochs',type=int,default=20); p.add_argument('--seed',type=int,default=42); args=p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    from transformers import AutoModelForCausalLM,AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model=AutoModelForCausalLM.from_pretrained(args.model_id,device_map='auto',torch_dtype=dtype)
    model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    df,labels=prepare_dataframe(args.csv_path); y=df.label_id.to_numpy(); train,val,test=stratified_split(y,args.seed)
    X=extract_representations(model,tokenizer,df.text.tolist(),args.batch_size,args.max_length); hidden=X.shape[1]
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); classifier=torch.nn.Linear(hidden,len(labels)).to(device); optimizer=torch.optim.AdamW(classifier.parameters(),lr=1e-3); loss_fn=torch.nn.CrossEntropyLoss(); best=None
    for epoch in range(args.epochs):
        classifier.train(); order=torch.randperm(len(train));
        for start in range(0,len(train),args.batch_size):
            idx=torch.tensor(train)[order[start:start+args.batch_size]]; loss=loss_fn(classifier(X[idx].to(device)),torch.tensor(y[idx.numpy()],device=device)); optimizer.zero_grad(); loss.backward(); optimizer.step()
        classifier.eval()
        with torch.no_grad(): pred=classifier(X[val].to(device)).argmax(1).cpu().numpy()
        metrics=classification_metrics(y[val],pred,len(labels));
        if best is None or metrics['macro_f1']>best['metrics']['macro_f1']: best={'metrics':metrics,'epoch':epoch+1,'state_dict':{k:v.cpu() for k,v in classifier.state_dict().items()}}
        print(f"epoch={epoch+1} val_accuracy={metrics['accuracy']:.4f} val_macro_f1={metrics['macro_f1']:.4f}")
    classifier.load_state_dict(best['state_dict']); classifier.eval()
    with torch.no_grad(): test_pred=classifier(X[test].to(device)).argmax(1).cpu().numpy()
    test_metrics=classification_metrics(y[test],test_pred,len(labels)); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':classifier.state_dict(),'hidden_size':hidden,'labels':labels,'model_id':args.model_id},out/'probe.pt')
    (out/'metrics.json').write_text(json.dumps({'model_id':args.model_id,'labels':labels,'seed':args.seed,'split_sizes':{'train':len(train),'validation':len(val),'test':len(test)},'best_epoch':best['epoch'],'validation':best['metrics'],'test':test_metrics},indent=2))
    print(json.dumps({'test_accuracy':test_metrics['accuracy'],'test_macro_f1':test_metrics['macro_f1'],'output_dir':str(out)},indent=2))
if __name__=='__main__': main()
