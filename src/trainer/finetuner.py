from datetime import datetime
import os
import time
import math
import mlflow
import torch
import psutil
import platform
from typing import Dict, Any
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

from src.config import FinetuneConfig, GPTConfig, LoggingConfig
from src.trainer import DataLoader, Logger
from src.model import GPT
from src.utils import load_checkpoint, get_system_info, get_memory_usage

class FineTuner:
    def __init__(self, checkpoint_path, finetune_config: FinetuneConfig, gpt_config: GPTConfig, device: str):
        self.finetune_config = finetune_config
        self.gpt_config = gpt_config
        self.device = device

        print(f"Loading model for finetuning from {checkpoint_path}")
        
        # Load checkpoint using utility function
        checkpoint, state_dict = load_checkpoint(checkpoint_path, device)
        self.checkpoint = checkpoint

        # Initialize the model first
        self.model = GPT(gpt_config).to(device)
        self.model.load_state_dict(state_dict)

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            target_modules=["c_attention", "c_projection"]
        )

        self.model = get_peft_model(self.model, lora_config)
        print(self.model.print_trainable_parameters())
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.finetune_config.learning_rate)

        # Initialize gradient scaler for mixed precision training
        self.scaler = torch.amp.GradScaler(enabled=(self.device == 'cuda'))

        # Load data - use os.path.join for cross-platform compatibility
        self.dataloader = DataLoader(
            data_dir=os.path.join('src', 'data', 'tinystories'), 
            block_size=self.gpt_config.block_size,
            batch_size=self.finetune_config.batch_size,
            device=self.device
        )

        # Create directory for saving finetune checkpoints    
        os.makedirs(self.finetune_config.checkpoint_dir, exist_ok=True)

        # Logger
        logging_config = LoggingConfig()
        self.logger = Logger(logging_config)
        
        # Track total tokens processed
        self.total_tokens_processed = 0
        
        # System info
        self.system_info = get_system_info(self.device)

    def _get_lora_stats(self) -> Dict[str, Any]:
        """Get LoRA-specific statistics."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params,
            'trainable_percentage': round((trainable_params / total_params) * 100, 4),
            'parameter_size_mb': round(trainable_params * 4 / (1024**2), 2),  # Assuming float32
        }

    def save_checkpoint(self, iter_num: int, loss: float, name: str = 'ckpt_finetune.pt', cleanup_after_logging: bool = False):
        # Save LoRA adapter weights
        adapter_path = os.path.join(self.finetune_config.checkpoint_dir, f"peft_{name}")
        print(f"saving checkpoint to {adapter_path}. Loss: {loss:.4f} at iteration {iter_num}")
        self.model.save_pretrained(adapter_path)
        
        # Log to MLflow
        print(f"Logging checkpoint and model to MLflow...")
        try:
            # Log the adapter directory as artifact (for resuming)
            self.logger.log_artifact(adapter_path, "lora_adapters")
            # Log the model (for deployment/versioning)
            self.logger.log_model(self.model, f"model_iter_{iter_num}")
            print(f"Checkpoint and model logged to MLflow")
            
            # Cleanup local directory if requested (for periodic checkpoints)
            if cleanup_after_logging:
                import shutil
                shutil.rmtree(adapter_path)
                print(f"Local checkpoint {adapter_path} deleted (stored in MLflow)")
        except Exception as e:
            print(f"Warning: Could not log to MLflow: {e}")

    def finetune_step(self):
        t0 = time.time() # Start timer
        for micro_step in range(self.finetune_config.gradient_accumulation_steps):
            X, Y = self.dataloader.get_batch('train')
            with torch.amp.autocast(device_type='cuda', enabled=(self.device == 'cuda')):
                logits, loss = self.model(X, Y)
                loss = loss / self.finetune_config.gradient_accumulation_steps # Normalize loss to account for gradient accumulation

            self.scaler.scale(loss).backward()
            
            # Track tokens processed
            self.total_tokens_processed += X.numel()

        # Calculate gradient norm before clipping
        grad_norm = None
        if self.finetune_config.grad_clip != 0.0:
            self.scaler.unscale_(self.optimizer) # unscale the gradients of optimizer's assigned params in-place
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.finetune_config.grad_clip)

        # Step the optimizer and scaler if training in fp16
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Flush the gradients as soon as we can, no need to wait for accumulation steps to finish
        self.optimizer.zero_grad(set_to_none=True)

        t1 = time.time() # End timer
        dt = t1 - t0

        return loss.item() * self.finetune_config.gradient_accumulation_steps, dt, grad_norm

    def _log_configurations(self):
        """Log all configurations and hyperparameters to MLflow."""
        # Finetuning configuration
        finetune_params = {
            'learning_rate': self.finetune_config.learning_rate,
            'num_epochs': self.finetune_config.num_epochs,
            'batch_size': self.finetune_config.batch_size,
            'gradient_accumulation_steps': self.finetune_config.gradient_accumulation_steps,
            'effective_batch_size': self.finetune_config.batch_size * self.finetune_config.gradient_accumulation_steps,
            'grad_clip': self.finetune_config.grad_clip,
            'checkpoint_interval': self.finetune_config.checkpoint_interval,
        }
        
        # Model configuration
        model_params = {
            'model_type': 'GPT-LoRA',
            'base_model': 'GPT',
            'num_layers': self.gpt_config.num_hidden_layers,
            'num_heads': self.gpt_config.num_attention_heads,
            'hidden_size': self.gpt_config.hidden_size,
            'vocab_size': self.gpt_config.vocab_size,
            'max_position_embeddings': self.gpt_config.max_position_embeddings,
            'dropout': self.gpt_config.dropout,
        }
        
        # LoRA configuration from the model
        lora_params = {
            'lora_r': 8,
            'lora_alpha': 16,
            'lora_dropout': 0.1,
            'lora_bias': 'none',
            'lora_target_modules': 'c_attention,c_projection',
        }
        
        # LoRA statistics
        lora_stats = self._get_lora_stats()
        
        # System info
        system_params = {f'system_{k}': v for k, v in self.system_info.items()}
        
        # Log all parameters
        all_params = {**finetune_params, **model_params, **lora_params, **lora_stats, **system_params}
        self.logger.log_params(all_params)
        
        # Log model architecture as text
        model_summary = str(self.model)
        self.logger.log_text(model_summary, 'model_architecture.txt')

    def finetune(self):
        start_finetune_time = time.time()
        print(f"Starting finetuning for {self.finetune_config.num_epochs} epochs... at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Prepare run tags
        run_tags = {
            'task': 'finetuning',
            'model_type': 'GPT-LoRA',
            'device': self.device,
            'dataset': 'tinystories',
            'framework': 'pytorch',
            'peft_method': 'LoRA',
        }
        
        run_name = f"gpt_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.start_run(run_name=run_name, tags=run_tags)
        
        # Log configurations
        self._log_configurations()

        self.best_loss = float('inf')
        self.iter_num = 0
        
        try:
            for epoch in range(self.finetune_config.num_epochs):
                print(f"Epoch {epoch+1}/{self.finetune_config.num_epochs}")
                start_epoch_time = time.time()
                total_epoch_loss = 0.0
                epoch_grad_norms = []
                
                for local_iter_num in range(self.dataloader.get_data_size('finetune') % (self.finetune_config.batch_size * self.finetune_config.gradient_accumulation_steps)):
                    current_loss, duration, grad_norm = self.finetune_step()
                    total_epoch_loss += current_loss
                    
                    if grad_norm is not None:
                        epoch_grad_norms.append(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)
                    
                    # Calculate tokens per second
                    tokens_per_sec = (self.finetune_config.batch_size * self.gpt_config.max_position_embeddings * 
                                      self.finetune_config.gradient_accumulation_steps) / duration
                    
                    # Calculate perplexity
                    perplexity = math.exp(current_loss) if current_loss < 10 else float('inf')

                    step_metrics = {
                        'step_loss': current_loss,
                        'step_perplexity': perplexity,
                        'step_lr': self.finetune_config.learning_rate,
                        'step_time_ms': duration * 1000,
                        'tokens_per_second': tokens_per_sec,
                    }
                    
                    if grad_norm is not None:
                        step_metrics['grad_norm'] = epoch_grad_norms[-1]
                    
                    self.logger.log_metrics(step_metrics, step=self.iter_num)
                    self.iter_num += 1

                end_epoch_time = time.time()
                epoch_duration = end_epoch_time - start_epoch_time
                print(f"Epoch {epoch+1} completed in {epoch_duration:.2f} seconds.")
                
                average_epoch_loss = total_epoch_loss / (local_iter_num + 1)
                average_perplexity = math.exp(average_epoch_loss) if average_epoch_loss < 10 else float('inf')
                
                epoch_metrics = {
                    'epoch_loss': average_epoch_loss,
                    'epoch_perplexity': average_perplexity,
                    'epoch_lr': self.finetune_config.learning_rate,
                    'epoch_time_seconds': epoch_duration,
                    'total_tokens_processed': self.total_tokens_processed,
                }
                
                # Add average gradient norm for the epoch
                if epoch_grad_norms:
                    epoch_metrics['epoch_avg_grad_norm'] = sum(epoch_grad_norms) / len(epoch_grad_norms)
                
                # Add memory usage
                epoch_metrics.update(get_memory_usage(self.device))
                
                self.logger.log_metrics(epoch_metrics, step=epoch)
                
                print(f"Epoch {epoch+1} average loss: {average_epoch_loss:.4f}, perplexity: {average_perplexity:.2f}")
                
                if average_epoch_loss < self.best_loss:
                    self.best_loss = average_epoch_loss
                    self.save_checkpoint(self.iter_num, self.best_loss, name='best_finetune_ckpt.pt')
                    
                    # Log the best checkpoint
                    best_ckpt_path = os.path.join(self.finetune_config.checkpoint_dir, 'peft_best_finetune_ckpt.pt')
                    if os.path.exists(best_ckpt_path):
                        self.logger.log_artifact(best_ckpt_path, "checkpoints")
                
                if (epoch + 1) % self.finetune_config.checkpoint_interval == 0:
                    self.save_checkpoint(self.iter_num, average_epoch_loss, name=f'epoch_{epoch+1}_finetune_ckpt.pt', cleanup_after_logging=True)

            end_finetune_time = time.time()
            total_duration = end_finetune_time - start_finetune_time
            
            # Log final metrics
            final_metrics = {
                'total_finetuning_time_hours': total_duration / 3600,
                'final_best_loss': self.best_loss,
                'final_best_perplexity': math.exp(self.best_loss) if self.best_loss < 10 else float('inf'),
                'total_epochs': self.finetune_config.num_epochs,
                'total_iterations': self.iter_num,
                'total_tokens_processed': self.total_tokens_processed,
            }
            self.logger.log_metrics(final_metrics, step=self.iter_num)
            
            # Log final LoRA model and adapter
            print("Logging final LoRA model and adapter to MLflow...")
            try:
                self.logger.log_model(self.model, "model_final")
                best_adapter_path = os.path.join(self.finetune_config.checkpoint_dir, 'peft_best_finetune_ckpt.pt')
                if os.path.exists(best_adapter_path):
                    self.logger.log_artifact(best_adapter_path, "lora_adapters")
            except Exception as e:
                print(f"Warning: Could not log final model to MLflow: {e}")
            
            print(f"Finetuning completed in {total_duration:.2f} seconds.")
            print(f"Best loss: {self.best_loss:.4f}")
            
            self.logger.end_run(status="FINISHED")
            
        except KeyboardInterrupt:
            print("\nFinetuning interrupted by user")
            self.logger.end_run(status="KILLED")
            raise
        except Exception as e:
            print(f"\nFinetuning failed with error: {e}")
            self.logger.end_run(status="FAILED")
            raise