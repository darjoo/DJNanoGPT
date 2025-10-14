import argparse
import torch
import tiktoken
from peft import LoraConfig, get_peft_model

from src.config import GPTConfig, TrainingConfig, FinetuneConfig
from src.model import GPT2
from src.trainer import Trainer, FineTuner

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='DJGPT - A GPT implementation')
    parser.add_argument('--cpu', action='store_true', 
                       help='Force CPU usage instead of GPU')
    parser.add_argument('--train', action='store_true', 
                       help='Flag to indicate training mode')
    parser.add_argument('--resume', type=str, default=None, 
                       help='Path to a checkpoint to resume training from')
    parser.add_argument('--infer', action='store_true', 
                       help='Flag to indicate inference mode')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to a model checkpoint for inference')
    parser.add_argument('--lora_ckpt', type=str, default=None,
                       help='Path to a LoRA checkpoint for inference')
    parser.add_argument('--text', type=str, default='',
                       help='Input text for inference')
    parser.add_argument('--finetune', action='store_true',
                       help='Flag to indicate finetuning mode')
    parser.add_argument('--finetune_ckpt', type=str, default=None,
                       help='Path to a checkpoint for finetuning')
    
    return parser.parse_args()

def validate_args(args: argparse.Namespace):
    if args.resume and not args.train:
        raise ValueError("--resume can only be used with --train flag")
    if args.infer and args.train:
        raise ValueError("--infer and --train flags cannot be used together")
    if not args.infer and not args.train and not args.finetune:
        raise ValueError("Either --infer or --train or --finetune flag must be provided")
    if args.infer and not args.text:
        raise ValueError("--text must be provided when using --infer flag")

def main():
    args = parse_args()
    validate_args(args)
    
    # You can now use args.cpu to check if the flag was provided
    if args.cpu:
        device = 'cpu'
    else:
        assert torch.cuda.is_available(), "CUDA is not available. Please run with --cpu flag."
        device = 'cuda'

    model_config = GPTConfig()
    model = GPT2(model_config).to(device)

    if args.train:
        training_config = TrainingConfig()
        trainer = Trainer(model, 
                          training_config, 
                          device, 
                          resume_checkpoint=args.resume if args.resume else None)

        trainer.train()
        print("Training complete.")
    elif args.infer:
        # Inference logic here
        print(f"Loading model for inference")
        torch.serialization.add_safe_globals([GPTConfig])
        checkpoint = torch.load(args.model, map_location=device)
        state_dict = checkpoint['model']

        # Fix keys of state dictionary
        unwanted_prefix = '_orig_mod.'
        for k in list(state_dict.keys()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

        model.load_state_dict(state_dict)

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            target_modules=["c_attention", "c_projection"]
        )

        model = get_peft_model(model, lora_config)
        if args.lora_ckpt:
            model.load_adapter(args.lora_ckpt, adapter_name="adapter1")

        # Inference
        print("Generating text...")
        enc = tiktoken.get_encoding("gpt2")
        sentence = args.text
        context = torch.tensor(enc.encode_ordinary(sentence)).unsqueeze(0).to(device)
        y = model.generate(context, max_new_tokens=200).squeeze().tolist()
        eot_pos = y.index(enc.eot_token) if enc.eot_token in y else len(y)
        decoded = enc.decode(y[:eot_pos])
        print(decoded)

        print("Inference complete.")
    elif args.finetune:
        if not args.finetune_ckpt:
            raise ValueError("--finetune_ckpt must be provided when using --finetune flag")
        
        finetune_config = FinetuneConfig()
        finetuner = FineTuner(args.finetune_ckpt, finetune_config, model_config, device)
        finetuner.finetune()
        print("Finetuning complete.")

if __name__ == "__main__":
    main()