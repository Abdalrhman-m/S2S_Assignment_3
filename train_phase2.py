# train_phase2.py - Phase 2 Training Script (Resume from Epoch 3 to Epoch 10)

import torch
import argparse
import time
from datetime import timedelta
import logging
import importlib
import os
from easydict import EasyDict as edict
import re

from lib.train.actors import SeqTrackActor
from lib.models.seqtrack import build_seqtrack
from lib.train.base_functions import build_dataloaders, get_optimizer_scheduler
from lib.train.admin.settings import Settings
from torch.nn import CrossEntropyLoss

from utils import set_seed, setup_logging, load_checkpoint


def save_checkpoint_phase2(epoch, model, optimizer, scheduler, local_path):
    """
    Save checkpoint locally only (no HuggingFace upload for Phase 2).
    """
    state = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'torch_rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    torch.save(state, local_path)
    logging.info(f"Checkpoint saved locally to {local_path}")


def main():
    parser = argparse.ArgumentParser(description="SeqTrack Assignment Training Script - Phase 2")
    parser.add_argument('--seed', type=int, default=2, help="Team number for the random seed (default: 2).")
    parser.add_argument('--epochs', type=int, default=10, help="Total number of epochs to train (default: 10).")
    parser.add_argument('--resume_from_epoch', type=int, default=3, help="Specify an epoch to resume from (default: 3).")
    parser.add_argument('--script', type=str, default='seqtrack', help='e.g., seqtrack')
    parser.add_argument('--config', type=str, default='seqtrack_b256', help="e.g., seqtrack_b256")
    parser.add_argument('--classes', nargs='+', default=['electricfan', 'mouse'], help='List of classes to train on.')
    args = parser.parse_args()

    settings = Settings()
    settings.script_name = args.script
    settings.config_name = args.config
    settings.env = edict()
    settings.env.lasot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'LaSOT_dataset')
    settings.use_lmdb = False
    
    # Phase 2 checkpoint directory
    settings.save_dir = os.path.join(os.getcwd(), "checkpoints_phase2")
    
    settings.local_rank = -1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    settings.device = device

    config_module = importlib.import_module(f"lib.config.{args.script}.config")
    cfg = config_module.cfg
    config_file_path = os.path.join(os.getcwd(), f"experiments/{args.script}/{args.config}.yaml")
    config_module.update_config_from_file(config_file_path)

    settings.search_area_factor = {'search': cfg.DATA.SEARCH.FACTOR, 'template': cfg.DATA.TEMPLATE.FACTOR}
    settings.output_sz = {'search': cfg.DATA.SEARCH.SIZE, 'template': cfg.DATA.TEMPLATE.SIZE}
    settings.center_jitter_factor = {'search': cfg.DATA.SEARCH.CENTER_JITTER, 'template': cfg.DATA.TEMPLATE.CENTER_JITTER}
    settings.scale_jitter_factor = {'search': cfg.DATA.SEARCH.SCALE_JITTER, 'template': cfg.DATA.TEMPLATE.SCALE_JITTER}
    settings.batchsize = cfg.TRAIN.BATCH_SIZE
    settings.grad_clip_norm = cfg.TRAIN.GRAD_CLIP_NORM
    settings.print_interval = cfg.TRAIN.PRINT_INTERVAL
    settings.scheduler_type = cfg.TRAIN.SCHEDULER.TYPE

    cfg.DATA.TRAIN.DATASETS_NAME = ["LASOT"]
    cfg.DATA.TRAIN.DATASETS_RATIO = [1]
    cfg.DATA.TRAIN.CLASSES = args.classes
    cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 7000

    if 'VAL' not in cfg.DATA:
        cfg.DATA.VAL = edict()
    cfg.DATA.VAL.SAMPLE_PER_EPOCH = 1000

    # Phase 2 log file
    logger = setup_logging('training_log_phase2.txt')

    logger.info("=" * 80)
    logger.info("PHASE 2 TRAINING - Resume from Epoch 3 to Epoch 10")
    logger.info("=" * 80)
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Total Epochs: {args.epochs}")
    logger.info(f"Resume from Epoch: {args.resume_from_epoch}")
    logger.info(f"Classes: {args.classes}")
    logger.info(f"Checkpoint Directory: {settings.save_dir}")
    logger.info(f"Log File: training_log_phase2.txt")
    logger.info("=" * 80)

    logger.info("--- Setting up model, data, and optimizer ---")
    loader_train, loader_val = build_dataloaders(cfg, settings)
    net = build_seqtrack(cfg)
    net.to(device)

    bins = cfg.MODEL.BINS
    weight = torch.ones(bins + 2).to(device)
    weight[bins] = 0.01
    weight[bins + 1] = 0.01
    objective = {'ce': CrossEntropyLoss(weight=weight)}
    loss_weight = {'ce': cfg.TRAIN.CE_WEIGHT}
    actor = SeqTrackActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    optimizer, scheduler = get_optimizer_scheduler(net, cfg)

    # Helper to find latest Phase 2 checkpoint
    def find_latest_phase2_checkpoint(save_dir: str):
        if not os.path.isdir(save_dir):
            return None
        latest = None
        latest_epoch_num = -1
        pattern = re.compile(r"epoch_(\d+)_phase2\.pth$")
        for fname in os.listdir(save_dir):
            m = pattern.match(fname)
            if not m:
                continue
            ep = int(m.group(1))
            if ep > latest_epoch_num:
                latest_epoch_num = ep
                latest = os.path.join(save_dir, fname)
        return latest

    # Prefer resuming from latest Phase 2 checkpoint if available; else fall back to Phase 1 resume
    start_epoch = 0
    latest_phase2_ckpt = find_latest_phase2_checkpoint(settings.save_dir)
    if latest_phase2_ckpt is not None:
        logger.info(f"Found latest Phase 2 checkpoint: {latest_phase2_ckpt}")
        start_epoch = load_checkpoint(net, optimizer, scheduler, latest_phase2_ckpt)
        logger.info(f"Successfully loaded Phase 2 checkpoint. Resuming from epoch {start_epoch}")
    elif args.resume_from_epoch is not None:
        checkpoint_epoch_to_load = args.resume_from_epoch - 1
        checkpoint_path = os.path.join(os.getcwd(), 'checkpoints', f'epoch_{checkpoint_epoch_to_load}.pth')
        logger.info(f"Loading Phase 1 checkpoint from: {checkpoint_path}")
        start_epoch = load_checkpoint(net, optimizer, scheduler, checkpoint_path)
        logger.info(f"Successfully loaded checkpoint. Resuming from epoch {start_epoch}")

    logger.info("--- Starting Phase 2 training ---")
    for epoch in range(start_epoch, args.epochs):
        # Reset RNG to a fixed seed (team number) at the beginning of each epoch
        # This aligns with the assignment requirement: fixed seed per epoch = team number
        set_seed(args.seed)
        actor.train()
        running_train_loss = 0.0
        epoch_start_time = time.time()
        last_log_time = time.time()

        for i, data in enumerate(loader_train, 1):
            data = data.to(device)
            try:
                loss, stats = actor(data)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item()
            except RuntimeError as e:
                # Attempt to recover from transient CUDA errors without killing the whole run
                if "CUDA" in str(e) or "cuda" in str(e):
                    logger.error(f"CUDA error on batch {i}: {e}. Attempting recovery: empty cache, skip batch.")
                    if torch.cuda.is_available():
                        try:
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                        except Exception:
                            pass
                    time.sleep(1)
                    continue
                else:
                    raise
            if i % settings.print_interval == 0:
                time_for_last_50 = time.time() - last_log_time
                time_since_beginning = time.time() - epoch_start_time
                samples_processed = i
                total_samples = len(loader_train)
                samples_left = total_samples - samples_processed
                time_per_sample = time_since_beginning / samples_processed if samples_processed > 0 else 0
                time_left_to_finish = samples_left * time_per_sample
                log_msg = (
                    f"Epoch {epoch + 1} : {samples_processed} / {total_samples} samples, "
                    f"time for last 50 samples: {str(timedelta(seconds=int(time_for_last_50)))} hours, "
                    f"time since beginning: {str(timedelta(seconds=int(time_since_beginning)))} hours, "
                    f"time left to finish the epoch: {str(timedelta(seconds=int(time_left_to_finish)))} hours"
                )
                logger.info(log_msg)
                last_log_time = time.time()

        avg_val_iou = 0.0
        if loader_val is not None:
            actor.eval()
            running_val_iou = 0.0
            with torch.no_grad():
                for data in loader_val:
                    data = data.to(device)
                    _, stats = actor(data)
                    running_val_iou += stats.get('IoU', 0)
            avg_val_iou = running_val_iou / len(loader_val)
        avg_train_loss = running_train_loss / len(loader_train)
        logger.info(f"Epoch {epoch + 1} Training Loss: {avg_train_loss:.4f}")
        logger.info(f"Epoch {epoch + 1} Validation IoU: {avg_val_iou:.4f}")
        scheduler.step()

        # Save checkpoint with phase2 naming
        local_path = os.path.join(settings.save_dir, f'epoch_{epoch}_phase2.pth')
        model_to_save = net.module if hasattr(net, 'module') else net
        save_checkpoint_phase2(epoch, model_to_save, optimizer, scheduler, local_path)
        # Free cached GPU memory between epochs to reduce chance of OOM / unknown errors
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        
    logger.info("=" * 80)
    logger.info("--- Phase 2 Training finished ---")
    logger.info("=" * 80)

if __name__ == '__main__':
    main()
