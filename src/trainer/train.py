"""
Training script

To run on a single GPU: 
$ python src/train.py --batch_size=32 --compile=False

To run on multiple GPUs, fx 4 GPUs on 1 node:
$ torchrun --standalone --nproc_per_node=4 src/train.py

Mlflow tracking:
$ mlflow ui --backend-store-uri ./mlflow_runs
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import mlflow
import mlflow.pytorch

from src.model import GPT
from src.config import GPTConfig

# ------------------------------------
# default config values
# I/O
out_dir = 'out'
eval_interval = 20
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume'

# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt' # 'wandb' or None
wandb_run_name = 'gpt' # 'run' or None

# mlflow logging
mlflow_tracking = True # enable MLflow tracking
mlflow_experiment_name = 'gpt-training' # experiment name
mlflow_tracking_uri = './mlflow_runs' # local directory for MLflow tracking

# data
dataset = 'tinystories'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 32 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024

# model
n_layer = 12
n_head = 12
n_embedding = 768
dropout = 0.0 # for pretraining 0.0 is good, for finetuning 0.1+ is better
bias = False # do we use bias inside LayerNorm and Linear layers?

# AdamW optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 6000000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0

# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 6000000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.

# System
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'mps'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# ------------------------------------

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging

# One gpu & process
master_process = True
seed_offset = 0
ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * batch_size * block_size * ddp_world_size
print(f"Tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)

# Initialize MLflow tracking
if mlflow_tracking and master_process:
    # Set tracking URI
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # Set experiment
    mlflow.set_experiment(mlflow_experiment_name)
    
    # Start MLflow run
    mlflow.start_run()
    
    # Log all configuration parameters
    mlflow.log_params(config)
    
    print(f"MLflow tracking started. Experiment: {mlflow_experiment_name}")
    print(f"MLflow UI available at: mlflow ui --backend-store-uri {mlflow_tracking_uri}")

torch.manual_seed(1337 + seed_offset) # ensure reproducibility
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# Float16 will automatically use GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Poor man's data loader
data_dir = os.path.join('src\\data', dataset)
def get_batch(split):
    """
    Generate a batch of data
    1. Recreate the memmap every time to prevent memory leak
    2. Randomly select batch_size starting indices for the sequences
    3. For each starting index, get the sequence of length block_size as input (x)
       and the sequence shifted by one as target (y)
    
    Args:
        split (str): 'train' or 'validation'
    
    Returns:
        x (torch.Tensor): Input tensor of shape (batch_size, block_size)
        y (torch.Tensor): Target tensor of shape (batch_size, block_size)
    """
    # Recreate memmap every batch to avoid memory leak
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'tinystories_train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'tinystories_validation.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+block_size+1]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # Pin arrays x & y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

iter_num = 0
best_val_loss = 1e9

# Attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} in {meta_path}")

# Initialize the model
model_args = dict(
    n_layer = n_layer,
    n_head = n_head,
    n_embedding = n_embedding,
    block_size = block_size,
    dropout = dropout,
    bias = bias,
    vocab_size = None,
)

if init_from == 'scratch':
    print("Initializing a new model from scratch")
    # Determine vocab_size
    if meta_vocab_size is None:
        print("Defaulting to vocab_size of 50304, since meta.pkl not found in data directory.")
    
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304

    # Create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # Resume from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    
    # Force config parameters to be the same as in the checkpoint
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        assert model_args[k] == checkpoint_model_args[k], f"for {k}, config file has {model_args[k]} but ckpt has {checkpoint_model_args[k]}"
    
    # Create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']

    # Fix keys of state dictionary
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    print(f"resumed from iteration {iter_num}, best val loss {best_val_loss:.4f}")
else:
    raise ValueError(f"init_from {init_from} not supported")

# Crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the model args reflect the true model
    print(f"Cropped model block size to {block_size}")
model.to(device)

# Initialize a GradScaler. If enabled=False, scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# Optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("optimizer state loaded from checkpoint")
checkpoint = None # free up memory

# Compile the model
if compile:
    print("compiling the model... (this may take a while)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0+
    print("model compiled")

@torch.no_grad()
def estimate_loss():
    """
    Estimate the loss on train and validation sets
    """
    out = {}
    model.eval()
    for split in ['train', 'validation']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(iteration):
    """
    Learning rate decay scheduler (cosine with warmup)

    Args:
        iteration (int): Current iteration number
    
    Returns:
        lr (float): Learning rate for the current iteration
    """
    # 1) linear warmup for warmup_iters steps
    if iteration < warmup_iters:
        return learning_rate * (iteration + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if iteration > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (iteration - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0.0 <= decay_ratio <= 1.0
    coefficient = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coefficient * (learning_rate - min_lr)

# Training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model
running_mfu = -1.0

while True:

    # Determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # Evaluate the loss on train/val sets and save the model checkpoint
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['validation']:.4f}, time {time.time()-t0:.2f}s")

        # Log metrics to MLflow
        if mlflow_tracking:
            mlflow.log_metrics({
                'train_loss': losses['train'].item(),
                'val_loss': losses['validation'].item(),
                'learning_rate': lr,
                'iteration': iter_num,
            }, step=iter_num)

        if losses['validation'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['validation']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                    'mlflow_experiment_name': mlflow.run.info.run_id
                }
                print(f"saving checkpoint to {out_dir}/ckpt.pt")
                ckpt_path = os.path.join(out_dir, 'ckpt.pt')
                torch.save(checkpoint, ckpt_path)
                
                # Log checkpoint as MLflow artifact
                if mlflow_tracking:
                    mlflow.log_artifact(ckpt_path, "checkpoints")
                    
                    # Also log the model using MLflow's pytorch integration
                    mlflow.pytorch.log_model(
                        pytorch_model=raw_model,
                        artifact_path="model",
                        registered_model_name=f"gpt-{dataset}" if iter_num % (eval_interval * 5) == 0 else None
                    )
    if iter_num == 0 and eval_only:
        print("eval_only mode, exiting")
        break

    # Forward backward update, with optional gradient accumulation to simulate larger batch size
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        
        # Immediately async prefetch the next batch while we are training (if device_type == 'cuda')
        X, Y = get_batch('train')

        # Backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    
    if grad_clip != 0.0:
        scaler.unscale_(optimizer) # unscale the gradients of optimizer's assigned params in-place
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # Step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()

    # Flush the gradients as soon as we can, no need to wait for accumulation steps to finish
    optimizer.zero_grad(set_to_none=True)

    # Timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # Get loss as float
        # Scale up to undo the normalization done during loss calculation, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, lr {lr:.2e}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        
        # Log step metrics to MLflow
        if mlflow_tracking:
            step_metrics = {
                'step_loss': lossf,
                'step_lr': lr,
                'step_time_ms': dt * 1000,
            }
            if running_mfu > 0:
                step_metrics['mfu_percent'] = running_mfu * 100
            mlflow.log_metrics(step_metrics, step=iter_num)
    
    iter_num += 1
    local_iter_num += 1 

    # Termination conditions
    if iter_num >= max_iters:
        print("max_iters reached, exiting")
        break

# End MLflow run
if mlflow_tracking and master_process:
    mlflow.end_run()
    print("MLflow run completed")