# DJGPT

A custom GPT (Generative Pre-trained Transformer) implementation with training, fine-tuning, and inference capabilities. This project includes support for LoRA (Low-Rank Adaptation) fine-tuning and uses MLflow for experiment tracking.

## Features

- **Training from scratch**: Train a GPT model on your own dataset
- **Fine-tuning with LoRA**: Efficiently adapt pre-trained models using Parameter-Efficient Fine-Tuning (PEFT)
- **Inference**: Generate text using trained or fine-tuned models
- **MLflow Integration**: Track experiments and metrics
- **Checkpoint Management**: Resume training and save/load models
- **GPU/CPU Support**: Flexible device selection

## Project Structure

```
DJGPT/
├── main.py                    # Main entry point
├── requirements.txt           # Python dependencies
├── checkpoints/              # Training checkpoints
├── finetune_checkpoints/     # Fine-tuning checkpoints (LoRA adapters)
├── mlruns/                   # MLflow experiment tracking
├── src/
│   ├── config/              # Configuration files
│   │   ├── gpt_config.py   # Model architecture config
│   │   ├── training_config.py
│   │   └── finetune_config.py
│   ├── model/               # Model implementation
│   │   ├── gpt.py
│   │   ├── block.py
│   │   ├── casual_self_attention.py
│   │   └── mlp.py
│   ├── trainer/             # Training and fine-tuning logic
│   │   ├── trainer.py
│   │   ├── finetuner.py
│   │   ├── dataloader.py
│   │   └── logger.py
│   └── utils/
└── test/                     # Unit tests
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, but recommended)

### Setup

1. Clone the repository:
```powershell
git clone https://github.com/darjoo/DJNanoGPT.git
cd DJGPT
```

2. Install dependencies:
```powershell
pip install -r requirements.txt
```

**Note**: The [requirements.txt](requirements.txt) includes PyTorch with CUDA 12.9 support. If you need a different CUDA version or CPU-only, modify the torch installation lines accordingly.

## Model Configuration

The default GPT configuration ([GPTConfig](src/config/gpt_config.py)) includes:
- **Context Length**: 256 tokens
- **Vocabulary Size**: 50,257 (GPT-2 tokenizer)
- **Layers**: 6
- **Attention Heads**: 6
- **Embedding Dimension**: 384
- **Dropout**: 0.1

You can modify these in [gpt_config.py](src/config/gpt_config.py).

## Usage

### 1. Training from Scratch

Train a new GPT model:
By default, uses CUDA.

```powershell
python main.py --train
```

**Training on CPU**:
```powershell
python main.py --train --cpu
```

**Resume training from a checkpoint**:
```powershell
python main.py --train --resume checkpoints/ckpt.pt
```

#### Training Configuration

Key training hyperparameters in [training_config.py](src/config/training_config.py)
Checkpoints are saved to the [checkpoints](checkpoints/) directory.

### 2. Fine-tuning with LoRA

Fine-tune a pre-trained model using LoRA adapters:

```powershell
python main.py --finetune --finetune_ckpt checkpoints/ckpt.pt
```

**Fine-tuning on CPU**:
```powershell
python main.py --finetune --finetune_ckpt checkpoints/ckpt.pt --cpu
```

#### Fine-tuning Configuration

Key fine-tuning hyperparameters in [finetune_config.py](src/config/finetune_config.py)

LoRA adapters are saved to [finetune_checkpoints](finetune_checkpoints/) with names like:
- `peft_epoch_1_finetune/`
- `peft_epoch_2_finetune/`
- `peft_best_finetune/` (best performing checkpoint)

### 3. Inference

Generate text using a trained model:

```powershell
python main.py --infer --model checkpoints/ckpt.pt --text "Once upon a time"
```

**Using a fine-tuned LoRA adapter**:
```powershell
python main.py --infer --model checkpoints/ckpt.pt --lora_ckpt finetune_checkpoints/peft_best_finetune --text "Once upon a time"
```

**Inference on CPU**:
```powershell
python main.py --infer --model checkpoints/ckpt.pt --text "Your prompt here" --cpu
```

#### Inference Parameters

- `--model`: Path to the base model checkpoint (required)
- `--lora_ckpt`: Path to LoRA adapter checkpoint (optional)
- `--text`: Input prompt for text generation (required)
- `--cpu`: Force CPU inference

The model generates up to 200 new tokens by default.

## Command-Line Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--cpu` | flag | Force CPU usage instead of GPU |
| `--train` | flag | Enable training mode |
| `--resume` | str | Path to checkpoint to resume training |
| `--infer` | flag | Enable inference mode |
| `--model` | str | Path to model checkpoint for inference |
| `--lora_ckpt` | str | Path to LoRA checkpoint for inference |
| `--text` | str | Input text prompt for inference |
| `--finetune` | flag | Enable fine-tuning mode |
| `--finetune_ckpt` | str | Path to checkpoint for fine-tuning |

## Examples

### Example 1: Complete Training Pipeline

```powershell
# Train a model from scratch
python main.py --train

# Fine-tune the trained model
python main.py --finetune --finetune_ckpt checkpoints/ckpt.pt

# Generate text with the fine-tuned model
python main.py --infer --model checkpoints/ckpt.pt --lora_ckpt finetune_checkpoints/peft_best_finetune --text "The future of AI is"
```

### Example 2: Quick Inference

```powershell
# Generate text with base model only
python main.py --infer --model checkpoints/ckpt.pt --text "In a world where"
```

### Example 3: Resume Interrupted Training

```powershell
# Training was interrupted, resume from last checkpoint
python main.py --train --resume checkpoints/ckpt.pt
```

## MLflow Experiment Tracking

The project uses MLflow to track training metrics, hyperparameters, and model artifacts. Training runs are logged to the `mlruns/` directory.

To view the MLflow UI:

```powershell
mlflow ui
```

Then navigate to `http://localhost:5000` in your browser.

## Testing

Run unit tests:

```powershell
pytest test/
```

## Architecture Details

### Model Components

- **GPT**: Main transformer model with decoder-only architecture
- **Block**: Transformer block with self-attention and feed-forward layers
- **CausalSelfAttention**: Multi-head self-attention with causal masking
- **MLP**: Feed-forward network with GELU activation

### Training Features

- **Gradient Accumulation**: Simulate large batch sizes on limited memory
- **Learning Rate Scheduling**: Warmup and cosine decay
- **Gradient Clipping**: Prevent exploding gradients
- **Model Compilation**: PyTorch 2.0 compilation for faster training
- **Checkpoint Management**: Automatic saving and resuming

### LoRA Fine-tuning

- **Target Modules**: Attention and projection layers
- **Rank (r)**: 8
- **Alpha**: 16
- **Dropout**: 0.1
- Memory-efficient adaptation with frozen base model

## Troubleshooting

### CUDA Not Available

If you see "CUDA is not available", either:
1. Install CUDA-compatible PyTorch
2. Use the `--cpu` flag

### Out of Memory

If you encounter OOM errors:
1. Reduce `batch_size` in config files
2. Increase `gradient_accumulation_steps` to maintain effective batch size
3. Reduce model size in `GPTConfig`
4. Use the `--cpu` flag (slower but uses system RAM)

### Import Errors

Ensure all dependencies are installed:
```powershell
pip install -r requirements.txt
```

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE)

## Acknowledgments

- Inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) - A minimal and educational GPT implementation
- Based on GPT architecture principles from OpenAI
- Uses Hugging Face's [PEFT](https://github.com/huggingface/peft) library for LoRA implementation
- Tokenization via tiktoken (GPT-2 encoding)

## Contact

Repository: [DJNanoGPT](https://github.com/darjoo/DJNanoGPT)

---

**Note**: This is an educational implementation. For production use cases, consider using established frameworks like Hugging Face Transformers.
