from datetime import datetime
import os
import time
import mlflow
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

from src.config import FinetuneConfig, GPTConfig, LoggingConfig
from src.trainer import DataLoader, Logger
from src.model import GPT

class FineTuner:
    def __init__(self, checkpoint_path, finetune_config: FinetuneConfig, gpt_config: GPTConfig, device: str):
        self.finetune_config = finetune_config
        self.gpt_config = gpt_config
        self.device = device

        print(f"Loading model for finetuning from {checkpoint_path}")
        torch.serialization.add_safe_globals([GPTConfig])
        self.checkpoint = torch.load(checkpoint_path, map_location=device)

        # Initialize the model first
        self.model = GPT(gpt_config).to(device)

        state_dict = self.checkpoint['model']

        # Fix keys of state dictionary
        unwanted_prefix = '_orig_mod.'
        for k in list(state_dict.keys()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

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

        # Load data
        self.dataloader = DataLoader(data_dir='src\\data\\tinystories', 
                                block_size=self.gpt_config.block_size,
                                batch_size=self.finetune_config.batch_size,
                                device=self.device)

        # Create directory for saving finetune checkpoints    
        os.makedirs(self.finetune_config.checkpoint_dir, exist_ok=True)

        # Logger
        logging_config = LoggingConfig()
        self.logger = Logger(logging_config)

    def save_checkpoint(self, iter_num: int, loss: float, name: str = 'ckpt_finetune.pt'):
        print(f"saving checkpoint to {self.finetune_config.checkpoint_dir}\\peft_{name}. Loss: {loss:.4f} at iteration {iter_num}")
        self.model.save_pretrained(os.path.join(self.finetune_config.checkpoint_dir, f"peft_{name}"))

    def finetune_step(self):
        t0 = time.time() # Start timer
        for micro_step in range(self.finetune_config.gradient_accumulation_steps):
            X, Y = self.dataloader.get_batch('train')
            with torch.amp.autocast(device_type='cuda', enabled=(self.device == 'cuda')):
                logits, loss = self.model(X, Y)
                loss = loss / self.finetune_config.gradient_accumulation_steps # Normalize loss to account for gradient accumulation

            self.scaler.scale(loss).backward()

        if self.finetune_config.grad_clip != 0.0:
            self.scaler.unscale_(self.optimizer) # unscale the gradients of optimizer's assigned params in-place
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.finetune_config.grad_clip)

        # Step the optimizer and scaler if training in fp16
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Flush the gradients as soon as we can, no need to wait for accumulation steps to finish
        self.optimizer.zero_grad(set_to_none=True)

        t1 = time.time() # End timer
        dt = t1 - t0

        return loss.item() * self.finetune_config.gradient_accumulation_steps, dt

    def finetune(self):
        start_finetune_time = time.time()
        print(f"Starting finetuning for {self.finetune_config.num_epochs} epochs... at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        self.best_loss = float('inf')
        self.iter_num = 0
        for epoch in range(self.finetune_config.num_epochs):
            print(f"Epoch {epoch+1}/{self.finetune_config.num_epochs}")
            start_epoch_time = time.time()
            total_epoch_loss = 0.0
            for local_iter_num in range(self.dataloader.get_data_size('finetune') % (self.finetune_config.batch_size * self.finetune_config.gradient_accumulation_steps)):
                current_loss, duration = self.finetune_step()
                total_epoch_loss += current_loss

                step_metrics = {
                    'step_loss': current_loss,
                    'step_lr': self.finetune_config.learning_rate,
                    'step_time_ms': duration * 1000,
                }
                self.logger.log_metrics(step_metrics, step=local_iter_num)
                self.iter_num += 1

            end_epoch_time = time.time()
            print(f"Epoch {epoch+1} completed in {end_epoch_time - start_epoch_time:.2f} seconds.")
            
            epoch_metrics = {
                'epoch_loss': total_epoch_loss,
                'epoch_lr': self.finetune_config.learning_rate,
                'epoch_time_ms': (end_epoch_time - start_epoch_time) * 1000,
            }
            self.logger.log_metrics(epoch_metrics, step=epoch)

            average_epoch_loss = total_epoch_loss / (local_iter_num + 1)
            print(f"Epoch {epoch+1} average loss: {average_epoch_loss:.4f}")
            if average_epoch_loss < self.best_loss:
                self.best_loss = average_epoch_loss
                self.save_checkpoint(self.iter_num, self.best_loss, name='best_finetune_ckpt.pt')
            
            if (epoch + 1) % self.finetune_config.checkpoint_interval == 0:
                self.save_checkpoint(self.iter_num, average_epoch_loss, name=f'epoch_{epoch+1}_finetune_ckpt.pt')

        end_finetune_time = time.time()
        print(f"Finetuning completed in {end_finetune_time - start_finetune_time:.2f} seconds.")