from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """
    Configuration for training a GPT model.
    
    Attributes:
        learning_rate (float): The initial learning rate for the optimizer.
        max_iters (int): The maximum number of training iterations.
        warmup_steps (int): The number of steps to linearly increase the learning rate.
        min_lr (float): The minimum learning rate after decay.
        eval_iters (int): The number of iterations between evaluations.
        batch_size (int): The number of sequences processed in one forward/backward pass.
        block_size (int): The context/sequence length.
        gradient_accumulation_steps (int): Number of steps to accumulate gradients before updating model
        weight_decay (float): Weight decay (L2 regularization) coefficient.
        beta1 (float): The beta1 parameter for the Adam optimizer.
        beta2 (float): The beta2 parameter for the Adam optimizer.
        eps (float): The epsilon parameter for the Adam optimizer.
        max_norm (float): Maximum norm for gradient clipping.
        
    Note:
        Effective batch size = batch_size x gradient_accumulation_steps
        Aim for effective batch size of 512-2048 for good performance    
    """

    learning_rate: float = 1e-4
    lr_decay_iters: int = 2000
    decay_lr = True # whether to decay the learning rate
    max_iters: int = 10000
    warmup_steps: int = 1000
    min_lr: float = 6e-5 
    eval_iters: int = 10
    batch_size: int = 32 # Number of sequences processed in one forward/backward pass
    block_size: int = 256 # Context/Sequence length
    gradient_accumulation_steps: int = 1 # Number of steps to accumulate gradients before updating model
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-9
    max_norm: float = 0.5
    grad_clip: float = 1.0 # Clip gradients at this value, or disable if == 0.0
    checkpoint_dir: str = 'checkpoints'
    compile: bool = True # Use PyTorch 2.0 to compile the model to be faster
    checkpoint_interval: int = 200

    def __post_init__(self):
        assert self.min_lr <= self.learning_rate, "min_lr should be less than or equal to learning_rate"
        assert self.max_iters > 0, "max_iters should be a positive integer"
        assert self.batch_size > 0, "batch_size should be a positive integer"
        assert self.block_size > 0, "block_size should be a positive integer"