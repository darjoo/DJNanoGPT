import math
import os 
import time
import torch
from datetime import datetime
from typing import Dict, Any

from src.config import GPTConfig
from src.model import GPT
from src.config import TrainingConfig, LoggingConfig
from src.utils import load_checkpoint, save_checkpoint, get_system_info, get_memory_usage
from .dataloader import DataLoader
from .logger import Logger

class Trainer:
    def __init__(self, model: GPT, training_config: TrainingConfig, device: str, resume_checkpoint: str = None):
        print(f"Training with the following config:\n{training_config}, device: {device}")

        self.model = model
        self.training_config = training_config
        self.device = device
        self.resume_checkpoint = resume_checkpoint

        self.init_optimizer()

        # Training state
        self.iter_num = 0
        self.best_val_loss = float('inf')
        self.log_run_id = None

        # Initialize the model
        self.model_args = dict(
            n_layer = model.config.num_hidden_layers,
            n_head = model.config.num_attention_heads,
            n_embedding = model.config.hidden_size,
            block_size = model.config.max_position_embeddings,
            dropout = model.config.dropout,
            bias = model.config.bias,
            vocab_size = None,
        )

        # Create checkpoint directory with error handling
        try:
            os.makedirs(self.training_config.checkpoint_dir, exist_ok=True)
        except OSError as e:
            raise RuntimeError(f"Failed to create checkpoint directory '{self.training_config.checkpoint_dir}': {e}")

        # Initialize gradient scaler for mixed precision training
        self.scaler = torch.amp.GradScaler(enabled=(self.device == 'cuda'))
        
        # Track total tokens processed
        self.total_tokens_processed = 0

        # Load data with configurable directory
        self.dataloader = DataLoader(data_dir=self.training_config.data_dir, 
                                block_size=self.training_config.block_size,
                                batch_size=self.training_config.batch_size,
                                device=self.device)
        
        # Logger
        logging_config = LoggingConfig()
        self.logger = Logger(logging_config)
        
        # System info
        self.system_info = get_system_info(self.device)

    def init_optimizer(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                            lr=self.training_config.learning_rate,
                                            betas=(self.training_config.beta1, self.training_config.beta2),
                                            eps=self.training_config.eps,
                                            weight_decay=self.training_config.weight_decay)

    def resume(self):
        print(f"Resuming training from checkpoint: {self.resume_checkpoint}")

        # Validate checkpoint file exists and load it
        checkpoint_path = os.path.join(self.training_config.checkpoint_dir, self.resume_checkpoint)
        
        # Load checkpoint with validation
        checkpoint, state_dict = load_checkpoint(
            checkpoint_path, 
            self.device
        )
        
        # Load model state
        self.model.load_state_dict(state_dict)
        self.iter_num = checkpoint['iter_num']
        self.best_val_loss = checkpoint['best_val_loss']
        self.log_run_id = checkpoint.get('wandb_run_id')
        
        # Restore scaler state if available
        if 'scaler' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler'])
        
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
        for _ in range(self.training_config.gradient_accumulation_steps):
            X, Y = self.dataloader.get_batch('train')
            with torch.amp.autocast(device_type='cuda', enabled=(self.device == 'cuda')):
                logits, loss = self.model(X, Y)
                loss = loss / self.training_config.gradient_accumulation_steps # Normalize loss to account for gradient accumulation

            self.scaler.scale(loss).backward()
            
            # Track tokens processed
            self.total_tokens_processed += X.numel()

        # Calculate gradient norm before clipping
        grad_norm = None
        if self.training_config.grad_clip != 0.0:
            self.scaler.unscale_(self.optimizer) # unscale the gradients of optimizer's assigned params in-place
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.grad_clip)

        # Step the optimizer and scaler if training in fp16
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Flush the gradients as soon as we can, no need to wait for accumulation steps to finish
        self.optimizer.zero_grad(set_to_none=True)

        t1 = time.time() # End timer
        dt = t1 - t0

        return loss.item() * self.training_config.gradient_accumulation_steps, dt, grad_norm
    
    @torch.no_grad()
    def estimate_loss(self):
        """
        Estimate the loss on train and validation sets.
        
        Returns:
            dict: Dictionary containing mean losses for 'train' and 'validation' splits
        """
        out = {}
        self.model.eval()
        for split in ['train', 'validation']:
            losses = torch.zeros(self.training_config.eval_steps)
            for k in range(self.training_config.eval_steps):
                X, Y = self.dataloader.get_batch(split)
                logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out
    
    def save_checkpoint(self, name: str = 'ckpt.pt'):
        """
        Save a training checkpoint.
        
        Args:
            name (str): Name of the checkpoint file
        """
        ckpt_path = os.path.join(self.training_config.checkpoint_dir, name)
        print(f"saving checkpoint to {ckpt_path}. Best val loss so far: {self.best_val_loss:.4f} at iteration {self.iter_num}")
        
        save_checkpoint(
            checkpoint_path=ckpt_path,
            model=self.model,
            optimizer=self.optimizer,
            scaler=self.scaler,
            model_args=self.model_args,
            iter_num=self.iter_num,
            best_val_loss=self.best_val_loss,
            config=self.model.config,
            run_id=self.logger.get_run_id()
        )

    def evaluate(self):
        losses = self.estimate_loss()
        
        # Calculate perplexity
        train_perplexity = math.exp(losses['train'].item())
        val_perplexity = math.exp(losses['validation'].item())

        metrics = {
            'eval_train_loss': losses['train'].item(),
            'eval_val_loss': losses['validation'].item(),
            'eval_train_perplexity': train_perplexity,
            'eval_val_perplexity': val_perplexity,
            'lr': self.get_lr(self.iter_num),
            'iteration': self.iter_num,
            'total_tokens_processed': self.total_tokens_processed,
        }
        
        # Add memory usage
        metrics.update(get_memory_usage(self.device))

        self.logger.log_metrics(metrics, step=self.iter_num)

        if losses['validation'] < self.best_val_loss:
            self.best_val_loss = losses['validation']
            self.save_checkpoint()

    def train(self):
        self.iter_num = 0
        self.best_val_loss = float('inf')

        if self.resume_checkpoint is not None:
            self.resume()

        # Prepare run tags
        run_tags = {
            'task': 'pretraining',
            'model_type': 'GPT',
            'device': self.device,
            'dataset': 'tinystories',
            'framework': 'pytorch',
            'resumed': 'true' if self.resume_checkpoint else 'false',
        }
        
        run_name = f"gpt_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.start_run(run_id=self.log_run_id, run_name=run_name, tags=run_tags)
        
        # Log configurations (only if not resuming, to avoid duplicates)
        if self.resume_checkpoint is None:
            self._log_configurations()

        # Compile the model
        if self.training_config.compile:
            print("compiling the model... (this may take a while)")
            self.model = torch.compile(self.model) # requires PyTorch 2.0+
            print("model compiled")

        # Training loop implementation
        start_time = time.time()
        print(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

        local_iter_num = 0
        
        try:
            while True:
                # Determine and set the learning rate for this iteration
                current_lr = self.get_lr(self.iter_num) if self.training_config.decay_lr else self.training_config.learning_rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = current_lr

                # Training step
                current_loss, duration, grad_norm = self.train_step()
                
                # Calculate tokens per second
                tokens_per_sec = (self.training_config.batch_size * self.training_config.block_size * 
                                  self.training_config.gradient_accumulation_steps) / duration
                
                # Calculate perplexity
                perplexity = math.exp(current_loss) if current_loss < 10 else float('inf')
                
                step_metrics = {
                    'step_loss': current_loss,
                    'step_perplexity': perplexity,
                    'step_lr': current_lr,
                    'step_time_ms': duration * 1000,
                    'tokens_per_second': tokens_per_sec,
                }
                
                # Add gradient norm if available
                if grad_norm is not None:
                    step_metrics['grad_norm'] = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
                
                self.logger.log_metrics(step_metrics, step=self.iter_num)

                # Evaluate at regular intervals (including at iteration 0 for baseline)
                if self.iter_num % self.training_config.eval_interval == 0:
                    self.evaluate()
                
                # Save checkpoint at intervals
                if self.iter_num % self.training_config.checkpoint_interval == 0 and self.iter_num > 0:
                    self.save_checkpoint(name=f'ckpt_iter_{self.iter_num}.pt')

                self.iter_num += 1
                local_iter_num += 1
                if self.iter_num >= self.training_config.max_iters:
                    break

            end_time = time.time()
            total_duration = end_time - start_time
            
            # Log final metrics
            final_metrics = {
                'total_training_time_hours': total_duration / 3600,
                'final_best_val_loss': self.best_val_loss,
                'total_iterations': self.iter_num,
                'total_tokens_processed': self.total_tokens_processed,
            }
            self.logger.log_metrics(final_metrics, step=self.iter_num)
            
            print(f"Training completed at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
            print(f"Total training time: {total_duration / 3600:.2f} hours")
            
            self.logger.end_run(status="FINISHED")
            
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            self.logger.end_run(status="KILLED")
            raise
        except Exception as e:
            print(f"\nTraining failed with error: {e}")
            self.logger.end_run(status="FAILED")
            raise
    
    def _log_configurations(self):
        """Log all configurations and hyperparameters to MLflow."""
        # Training configuration
        training_params = {
            'learning_rate': self.training_config.learning_rate,
            'max_iters': self.training_config.max_iters,
            'batch_size': self.training_config.batch_size,
            'block_size': self.training_config.block_size,
            'gradient_accumulation_steps': self.training_config.gradient_accumulation_steps,
            'effective_batch_size': self.training_config.batch_size * self.training_config.gradient_accumulation_steps,
            'warmup_steps': self.training_config.warmup_steps,
            'lr_decay_iters': self.training_config.lr_decay_iters,
            'min_lr': self.training_config.min_lr,
            'weight_decay': self.training_config.weight_decay,
            'beta1': self.training_config.beta1,
            'beta2': self.training_config.beta2,
            'grad_clip': self.training_config.grad_clip,
            'eval_interval': self.training_config.eval_interval,
            'eval_steps': self.training_config.eval_steps,
            'compile': self.training_config.compile,
        }
        
        # Model configuration
        model_params = {
            'model_type': 'GPT',
            'num_layers': self.model.config.num_hidden_layers,
            'num_heads': self.model.config.num_attention_heads,
            'hidden_size': self.model.config.hidden_size,
            'vocab_size': self.model.config.vocab_size,
            'max_position_embeddings': self.model.config.max_position_embeddings,
            'dropout': self.model.config.dropout,
            'bias': self.model.config.bias,
        }
        
        # Calculate total parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        model_params['total_parameters'] = total_params
        model_params['trainable_parameters'] = trainable_params
        model_params['parameter_size_mb'] = round(total_params * 4 / (1024**2), 2)  # Assuming float32
        
        # System info
        system_params = {f'system_{k}': v for k, v in self.system_info.items()}
        
        # Log all parameters
        all_params = {**training_params, **model_params, **system_params}
        self.logger.log_params(all_params)
        
        # Log model architecture as text
        model_summary = str(self.model)
        self.logger.log_text(model_summary, 'model_architecture.txt')
