import argparse
import torch
import tiktoken

from src.config import GPTConfig, TrainingConfig
from src.model import GPT
from src.trainer import Trainer

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='DJGPT - A GPT implementation')
    parser.add_argument('--cpu', action='store_true', 
                       help='Force CPU usage instead of GPU')
    parser.add_argument('--train', action='store_true', 
                       help='Flag to indicate training mode')
    parser.add_argument('--resume', action='store_true', 
                       help='Path to a checkpoint to resume training from')
    parser.add_argument('--infer', action='store_true', 
                       help='Flag to indicate inference mode')
    parser.add_argument('--text', type=str, default='',
                       help='Input text for inference')
    
    return parser.parse_args()

def validate_args(args: argparse.Namespace):
    if args.resume and not args.train:
        raise ValueError("--resume can only be used with --train flag")
    if args.infer and args.train:
        raise ValueError("--infer and --train flags cannot be used together")
    if not args.infer and not args.train:
        raise ValueError("Either --infer or --train flag must be provided")
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
    model = GPT(model_config).to(device)

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
        checkpoint = torch.load('model_checkpoint.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Inference
        print("Generating text...")
        enc = tiktoken.get_encoding("gpt2")
        sentence = args.text
        context = torch.tensor(enc.encode_ordinary(sentence)).unsqueeze(0).to(device)
        y = model.generate(context, max_new_tokens=200)
        print(enc.decode(y.squeeze().tolist()))

        print("Inference complete.")

if __name__ == "__main__":
    main()