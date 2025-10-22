# train.py

import torch
import argparse
import time
from datetime import timedelta
import logging
import importlib
import os
from easydict import EasyDict as edict

from lib.train.actors import SeqTrackActor
from lib.models.seqtrack import build_seqtrack
from lib.train.base_functions import build_dataloaders, get_optimizer_scheduler
from lib.train.admin.settings import Settings
from torch.nn import CrossEntropyLoss

from utils import set_seed, setup_logging, save_checkpoint, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="SeqTrack Assignment Training Script")
    parser.add_argument('--seed', type=int, required=True, help="Team number for the random seed.")
    parser.add_argument('--epochs', type=int, required=True, help="Total number of epochs to train.")
    parser.add_argument('--resume_from_epoch', type=int, default=None, help="Specify an epoch to resume from.")
    parser.add_argument('--hf_repo_id', type=str, required=True, help="Hugging Face repo ID.")
    parser.add_argument('--script', type=str, required=True, help='e.g., seqtrack')
    parser.add_argument('--config', type=str, required=True, help="e.g., seqtrack_b256")
    parser.add_argument('--classes', nargs='+', required=True, help='List of classes to train on (e.g., electricfan mouse).')
    args = parser.parse_args()

    settings = Settings()
    settings.script_name = args.script
    settings.config_name = args.config
    settings.env = edict()
    settings.env.lasot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'LaSOT_dataset')
    settings.use_lmdb = False
    settings.save_dir = os.path.join(os.getcwd(), "checkpoints")
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

    logger = setup_logging('training_log.txt')

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

    start_epoch = 0
    if args.resume_from_epoch is not None:
        checkpoint_epoch_to_load = args.resume_from_epoch - 1
        checkpoint_path = os.path.join(settings.save_dir, f'epoch_{checkpoint_epoch_to_load}.pth')
        start_epoch = load_checkpoint(net, optimizer, scheduler, checkpoint_path)

    logger.info("--- Starting training ---")
    for epoch in range(start_epoch, args.epochs):
        set_seed(args.seed + epoch)
        actor.train()
        running_train_loss = 0.0
        epoch_start_time = time.time()
        last_log_time = time.time()

        for i, data in enumerate(loader_train, 1):
            data = data.to(device)
            loss, stats = actor(data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
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

        local_path = os.path.join(settings.save_dir, f'epoch_{epoch}.pth')
        hf_path = f'epoch_{epoch}.pth'
        model_to_save = net.module if hasattr(net, 'module') else net
        save_checkpoint(epoch, model_to_save, optimizer, scheduler, local_path, args.hf_repo_id, hf_path)
    logger.info("--- Training finished ---")

if __name__ == '__main__':
    main()