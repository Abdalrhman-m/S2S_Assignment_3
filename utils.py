# utils.py

import torch
import random
import numpy as np
import os
import logging
import time
from huggingface_hub import HfApi, HfFolder

# Set to False to skip HuggingFace uploads and save time
ENABLE_HF_UPLOAD = False


def set_seed(seed):
    """
    Sets the random seed for reproducibility for all relevant libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logging(log_file='training_log.txt'):
    """
    Configures logging to print to both the console and a file.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()


def save_checkpoint(epoch, model, optimizer, scheduler, local_path, hf_repo_id, hf_path):
    """
    Saves a complete checkpoint and uploads it to Hugging Face Hub.

    The checkpoint includes:
    - Model state_dict
    - Optimizer state_dict
    - Scheduler state_dict
    - RNG states (Python, NumPy, PyTorch)
    - Current epoch
    """
    # 1. Ensure the local directory exists
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # 2. Gather all states into a dictionary
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'rng_states': {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        }
    }

    # 3. Save the checkpoint locally
    torch.save(checkpoint, local_path)
    logging.info(f"Checkpoint saved locally at {local_path}")

    # 4. Upload to Hugging Face Hub (with better error handling)
    if ENABLE_HF_UPLOAD:
        try:
            logging.info(f"Attempting to upload checkpoint to Hugging Face repo: {hf_repo_id}")
            api = HfApi()
            
            # The token should already be authenticated via huggingface-cli login
            # But we can also try to read it from the token file as backup
            token = None
            try:
                from huggingface_hub import HfFolder
                token = HfFolder.get_token()
            except:
                pass
            
            # If no token found, try reading from file
            if token is None:
                token_file = os.path.join(os.path.dirname(__file__), 'Hugging face token.txt')
                if os.path.exists(token_file):
                    with open(token_file, 'r') as f:
                        content = f.read()
                        import re
                        match = re.search(r'hf_[A-Za-z0-9]+', content)
                        if match:
                            token = match.group(0)
            
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=hf_path,
                repo_id=hf_repo_id,
                repo_type="model",
                token=token
            )
            logging.info(f"✅ Checkpoint successfully uploaded to Hugging Face repo: {hf_repo_id}")
        except Exception as e:
            logging.warning(f"⚠️  Failed to upload checkpoint to Hugging Face: {e}")
            logging.warning("Continuing training without HuggingFace upload...")
    else:
        logging.info(f"ℹ️  HuggingFace upload skipped (ENABLE_HF_UPLOAD=False). Checkpoint saved locally only.")


def load_checkpoint(model, optimizer, scheduler, filepath):
    """
    Loads a complete checkpoint from a file to resume training.

    Supports both Phase 1 (original) and Phase 2 checkpoint formats.
    Phase 1 keys:
      - 'model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict'
      - 'rng_states': {'python', 'numpy', 'torch', 'cuda'}
      - 'epoch' (0-based index of last finished epoch)
    Phase 2 keys:
      - 'state_dict', 'optimizer', 'scheduler'
      - 'torch_rng_state', 'cuda_rng_state'
      - 'epoch' (already the start epoch to resume from)
    """
    if not os.path.exists(filepath):
        logging.warning(f"Checkpoint file not found at {filepath}. Starting from scratch.")
        return 0  # Start epoch

    checkpoint = torch.load(filepath, weights_only=False)

    # Detect format and load states
    if 'model_state_dict' in checkpoint:
        # Phase 1 format
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Restore RNG states
        rng = checkpoint.get('rng_states', {})
        py_state = rng.get('python')
        np_state = rng.get('numpy')
        torch_state = rng.get('torch')
        cuda_state = rng.get('cuda')
        if py_state is not None:
            random.setstate(py_state)
        if np_state is not None:
            np.random.set_state(np_state)
        if torch_state is not None:
            torch.set_rng_state(torch_state)
        if torch.cuda.is_available() and cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state)

        # Start from next epoch after the saved one
        start_epoch = int(checkpoint.get('epoch', -1)) + 1
    else:
        # Phase 2 format
        model.load_state_dict(checkpoint['state_dict'])
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])

        # Restore RNG states (optional)
        torch_state = checkpoint.get('torch_rng_state')
        cuda_state = checkpoint.get('cuda_rng_state')
        if torch_state is not None:
            torch.set_rng_state(torch_state)
        if torch.cuda.is_available() and cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state)

        # In Phase 2 we saved 'epoch' as the exact resume epoch
        start_epoch = int(checkpoint.get('epoch', 0))

    logging.info(f"Successfully loaded checkpoint. Resuming from epoch {start_epoch}")
    return start_epoch