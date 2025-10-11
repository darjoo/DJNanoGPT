import math
import mlflow
import mlflow.pytorch
import os 
import time
import torch

from src.config import GPTConfig
from src.model import GPT
from src.config import TrainingConfig, LoggingConfig
from .dataloader import DataLoader
from .logger import Logger

class Trainer:
    def __init__(self, model: GPT, training_config: TrainingConfig, device: str, resume_checkpoint: str = None):
        self.model = model
        self.training_config = training_config
        self.device = device
        self.resume_checkpoint = resume_checkpoint
        self.best_loss = float('inf')

        self.init_optimizer()

        # Training state
        self.iter_num = 0
        self.best_val_loss = float('inf')
        self.current_iter = 0
        self.train_losses = []
        self.val_losses = []
        self.log_run_id = None

        # Initialize the model
        self.model_args = dict(
            n_layer = model.config.n_layer,
            n_head = model.config.n_head,
            n_embedding = model.config.n_embedding,
            block_size = model.config.block_size,
            dropout = model.config.dropout,
            bias = model.config.bias,
            vocab_size = None,
        )

        # Create checkpoint directory
        os.makedirs(self.training_config.checkpoint_dir, exist_ok=True)

        # Initialize gradient scaler for mixed precision training
        self.scaler = torch.amp.GradScaler(enabled=(self.device == 'cuda'))

        # Initialize training metrics
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'grad_norm': [],
            'memory_usage': []
        }

        # Load data
        self.dataloader = DataLoader(data_dir='src\\data\\tinystories', 
                                block_size=self.training_config.block_size,
                                batch_size=self.training_config.batch_size,
                                device=self.device)
        
        # Logger
        logging_config = LoggingConfig()
        self.logger = Logger(logging_config, self.training_config)

    def init_optimizer(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                            lr=self.training_config.learning_rate,
                                            betas=(self.training_config.beta1, self.training_config.beta2),
                                            eps=self.training_config.eps,
                                            weight_decay=self.training_config.weight_decay)

    def resume(self):
        print(f"Resuming training from checkpoint: {self.resume_checkpoint}")

        torch.serialization.add_safe_globals([GPTConfig])

        checkpoint = torch.load(f"{os.path.join(self.training_config.checkpoint_dir, self.resume_checkpoint)}", map_location=self.device)
        checkpoint_model_args = checkpoint['model_args']

        # Force config parameters to be the same as in the checkpoint
        for k in ['n_layer', 'n_head', 'n_embedding', 'block_size', 'bias', 'vocab_size']:
            assert self.model_args[k] == checkpoint_model_args[k], f"for {k}, config file has {self.model_args[k]} but ckpt has {checkpoint_model_args[k]}"
        
        state_dict = checkpoint['model']

        # Fix keys of state dictionary
        unwanted_prefix = '_orig_mod.'
        for k in list(state_dict.keys()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
        self.model.load_state_dict(state_dict)
        self.iter_num = checkpoint['iter_num']
        self.best_val_loss = checkpoint['best_val_loss']
        self.log_run_id = checkpoint['mlflow_experiment_name']
        print(f"resumed from iteration {self.iter_num}, best val loss {self.best_val_loss:.4f}")

    def get_lr(self, iteration):
        """
        Learning rate decay scheduler (cosine with warmup)

        Args:
            iteration (int): Current iteration number
        
        Returns:
            lr (float): Learning rate for the current iteration
        """
        # 1) linear warmup for warmup_iters steps
        if iteration < self.training_config.warmup_steps:
            return self.training_config.learning_rate * (iteration + 1) / (self.training_config.warmup_steps + 1)
        # 2) if it > lr_decay_iters, return min learning rate
        if iteration > self.training_config.lr_decay_iters:
            return self.training_config.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (iteration - self.training_config.warmup_steps) / (self.training_config.lr_decay_iters - self.training_config.warmup_steps)
        assert 0.0 <= decay_ratio <= 1.0
        coefficient = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return self.training_config.min_lr + coefficient * (self.training_config.learning_rate - self.training_config.min_lr)

    def train_step(self):
        t0 = time.time() # Start timer
        for micro_step in range(self.training_config.gradient_accumulation_steps):
            X, Y = self.dataloader.get_batch('train')
            with torch.amp.autocast(device_type='cuda', enabled=(self.device == 'cuda')):
                logits, loss = self.model(X, Y)
                loss = loss / self.training_config.gradient_accumulation_steps # Normalize loss to account for gradient accumulation

            self.scaler.scale(loss).backward()

        if self.training_config.grad_clip != 0.0:
            self.scaler.unscale_(self.optimizer) # unscale the gradients of optimizer's assigned params in-place
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.grad_clip)

        # Step the optimizer and scaler if training in fp16
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Flush the gradients as soon as we can, no need to wait for accumulation steps to finish
        self.optimizer.zero_grad(set_to_none=True)

        t1 = time.time() # End timer
        dt = t1 - t0

        return loss.item() * self.training_config.gradient_accumulation_steps, dt
    
    @torch.no_grad()
    def estimate_loss(self):
        """
        Estimate the loss on train and validation sets
        """
        out = {}
        self.model.eval()
        for split in ['train', 'validation']:
            losses = torch.zeros(self.training_config.eval_iters)
            for k in range(self.training_config.eval_iters):
                X, Y = self.dataloader.get_batch(split)
                with torch.no_grad():
                    logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out
    
    def save_checkpoint(self, name: str = 'ckpt.pt'):
        checkpoint = {
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'model_args': self.model_args,
                    'iter_num': self.iter_num,
                    'best_val_loss': self.best_val_loss,
                    'config': self.model.config,
                    'mlflow_experiment_name': mlflow.active_run().info.run_id
                }
        ckpt_path = os.path.join(self.training_config.checkpoint_dir, name)
        print(f"saving checkpoint to {ckpt_path}. Best val loss so far: {self.best_val_loss:.4f} at iteration {self.iter_num}")
        torch.save(checkpoint, ckpt_path)

    def evaluate(self):
        losses = self.estimate_loss()

        metrics = {
            'eval_train_loss': losses['train'].item(),
            'eval_val_loss': losses['validation'].item(),
            'lr': self.get_lr(self.iter_num),
            'iteration': self.iter_num,
        }

        self.logger.log_metrics(metrics, step=self.iter_num)

        self.val_losses.append(losses['validation'])

        if losses['validation'] < self.best_val_loss:
            self.best_val_loss = losses['validation']
            self.save_checkpoint()

    def train(self):
        self.iter_num = 0
        self.best_val_loss = float('inf')

        if self.resume_checkpoint is not None:
            self.resume()

        self.logger.start_run(self.log_run_id)

        # Compile the model
        if self.training_config.compile:
            print("compiling the model... (this may take a while)")
            self.model = torch.compile(self.model) # requires PyTorch 2.0+
            print("model compiled")

        # Training loop implementation
        print(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

        current_loss = self.train_step()
        local_iter_num = 0

        while True:
            # Determine and set the learning rate for this iteration
            current_lr = self.get_lr(self.iter_num) if self.training_config.decay_lr else self.training_config.learning_rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr

            # Training step
            current_loss, duration = self.train_step()
            
            step_metrics = {
                'step_loss': current_loss,
                'step_lr': current_lr,
                'step_time_ms': duration * 1000,
            }
            self.logger.log_metrics(step_metrics, step=self.iter_num)

            if local_iter_num != 0 and self.iter_num % self.training_config.eval_iters == 0:
                self.evaluate()

            self.iter_num += 1
            local_iter_num += 1
            if self.iter_num >= self.training_config.max_iters:
                break

        print(f"Training completed at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")