import math
import mlflow
import mlflow.pytorch
import os 
import time
import torch
import psutil
import platform
from datetime import datetime
from typing import Dict, Any

from src.config import GPTConfig
from src.model import GPT
from src.config import TrainingConfig, LoggingConfig
from .dataloader import DataLoader
from .logger import Logger

class Trainer:
    def __init__(self, model: GPT, training_config: TrainingConfig, device: str, resume_checkpoint: str = None):
        print(f"Training with the following config:\n{training_config}, device: {device}")

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
            n_layer = model.config.num_hidden_layers,
            n_head = model.config.num_attention_heads,
            n_embedding = model.config.hidden_size,
            block_size = model.config.max_position_embeddings,
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
        
        # Track total tokens processed
        self.total_tokens_processed = 0

        # Load data
        self.dataloader = DataLoader(data_dir='src\\data\\tinystories', 
                                block_size=self.training_config.block_size,
                                batch_size=self.training_config.batch_size,
                                device=self.device)
        
        # Logger
        logging_config = LoggingConfig()
        self.logger = Logger(logging_config)
        
        # System info
        self.system_info = self._get_system_info()

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
    
    def save_checkpoint(self, name: str = 'ckpt.pt', cleanup_after_logging: bool = False):
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
        
        # Log to MLflow
        print(f"Logging checkpoint and model to MLflow...")
        try:
            # Log the checkpoint file as artifact (for resuming)
            self.logger.log_artifact(ckpt_path, "checkpoints")
            # Log the model (for deployment/versioning)
            self.logger.log_model(self.model, f"model_iter_{self.iter_num}")
            print(f"Checkpoint and model logged to MLflow")
            
            # Cleanup local file if requested (for periodic checkpoints)
            if cleanup_after_logging:
                os.remove(ckpt_path)
                print(f"Local checkpoint {ckpt_path} deleted (stored in MLflow)")
        except Exception as e:
            print(f"Warning: Could not log to MLflow: {e}")

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
        metrics.update(self._get_memory_usage())

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

                if local_iter_num != 0 and self.iter_num % self.training_config.eval_iters == 0:
                    self.evaluate()
                
                # Save and log checkpoint at intervals
                if self.iter_num % self.training_config.checkpoint_interval == 0 and self.iter_num > 0:
                    self.save_checkpoint(name=f'ckpt_iter_{self.iter_num}.pt', cleanup_after_logging=True)

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
            
            # Log final model and checkpoint
            print("Logging final model and checkpoint to MLflow...")
            try:
                self.logger.log_model(self.model, "model_final")
                best_ckpt = os.path.join(self.training_config.checkpoint_dir, 'ckpt.pt')
                if os.path.exists(best_ckpt):
                    self.logger.log_artifact(best_ckpt, "checkpoints")
            except Exception as e:
                print(f"Warning: Could not log final model to MLflow: {e}")
            
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

    def _get_system_info(self) -> Dict[str, Any]:
        """Collect system information for logging."""
        info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'pytorch_version': torch.__version__,
            'device': self.device,
            'cpu_count': psutil.cpu_count(),
            'total_ram_gb': round(psutil.virtual_memory().total / (1024**3), 2)
        }
        
        if self.device == 'cuda':
            info['cuda_version'] = torch.version.cuda
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory_gb'] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        
        return info
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        memory_stats = {}
        
        # CPU memory
        memory_stats['cpu_memory_used_gb'] = round(psutil.virtual_memory().used / (1024**3), 2)
        memory_stats['cpu_memory_percent'] = psutil.virtual_memory().percent
        
        # GPU memory
        if self.device == 'cuda':
            memory_stats['gpu_memory_allocated_gb'] = round(torch.cuda.memory_allocated() / (1024**3), 2)
            memory_stats['gpu_memory_reserved_gb'] = round(torch.cuda.memory_reserved() / (1024**3), 2)
            memory_stats['gpu_memory_percent'] = round(
                (torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory) * 100, 2
            )
        
        return memory_stats
    
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
            'eval_iters': self.training_config.eval_iters,
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
