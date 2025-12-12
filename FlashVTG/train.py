import os
import time
import json
import pprint
import random
import numpy as np
from tqdm import tqdm, trange
from collections import defaultdict

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from FlashVTG.config import BaseOptions
from FlashVTG.start_end_dataset import (
    StartEndDataset,
    start_end_collate,
    prepare_batch_inputs,
)
from FlashVTG.inference import eval_epoch, start_inference, setup_model, setup_model_with_gan
from FlashVTG.gan_model import wgan_gp_mse_loss, feature_matching_loss
from utils.basic_utils import AverageMeter, dict_to_markdown
from gan_model import wgan_gp_mse_loss, feature_matching_loss, gaussian_mask_1d

import nncore
from datetime import datetime
import logging


def set_seed(seed, use_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i,
                tb_writer):
    logger.info(f"[Epoch {epoch_i+1}]")
    model.train()

    # init meters
    loss_meters = defaultdict(AverageMeter)

    num_training_examples = len(train_loader)

    # iteration loop
    timer_dataloading = time.time()
    for batch_idx, batch in tqdm(enumerate(train_loader),
                                 desc="Training Iteration",
                                 total=num_training_examples):

        model_inputs, targets = prepare_batch_inputs(
            batch[1], opt.device, non_blocking=opt.pin_memory)

        targets["label"] = batch[0]
        targets["fps"] = torch.full((256, ), 1 / opt.clip_length).to(
            opt.device)  # if datasets is qv, fps is 0.5
        outputs = model(**model_inputs, targets=targets)

        loss_dict = criterion(batch, outputs, targets)
        loss_dict = {k: v for k, v in outputs.items() if 'loss' in k}

        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys()
                     if k in weight_dict)

        if torch.isnan(losses).any():
            print("Loss contains NaN values")

        optimizer.zero_grad()
        losses.backward()

        if opt.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(),
                                     opt.grad_clip,
                                     error_if_nonfinite=False)
        optimizer.step()

        loss_dict["weighted_loss_overall"] = float(losses)  # for logging only
        for k, v in loss_dict.items():
            loss_meters[k].update(float(v))

        # Output and log loss info every iteration
        current_loss = {k: v.avg for k, v in loss_meters.items()}
        for k, v in current_loss.items():
            tb_writer.add_scalar(f"Train/{k}", v,
                                 epoch_i * num_training_examples + batch_idx)

        tb_writer.add_scalar("Train/lr",
                             float(optimizer.param_groups[0]["lr"]),
                             epoch_i * num_training_examples + batch_idx)

    # Write epoch-level logs to file
    to_write = opt.train_log_txt_formatter.format(
        time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
        epoch=epoch_i + 1,
        loss_str=" ".join(
            ["{} {:.4f}".format(k, v.avg) for k, v in loss_meters.items()]),
    )
    logger.info(to_write)
    with open(opt.train_log_filepath, "a") as f:
        f.write(to_write)

    return losses, epoch_i * num_training_examples + batch_idx


def train(model, criterion, optimizer, lr_scheduler, train_dataset,
          val_dataset, opt):
    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)

    tb_writer = SummaryWriter(opt.tensorboard_log_dir)
    tb_writer.add_text("hyperparameters",
                       dict_to_markdown(vars(opt), max_str_len=None))
    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"

    train_loader = DataLoader(train_dataset,
                              collate_fn=start_end_collate,
                              batch_size=opt.bsz,
                              num_workers=opt.num_workers,
                              shuffle=True,
                              pin_memory=opt.pin_memory)

    prev_best_score = 0.0
    es_cnt = 0  # early stop counter
    if opt.start_epoch is None:
        start_epoch = -1 if opt.eval_untrained else 0
    else:
        start_epoch = opt.start_epoch
    save_submission_filename = "latest_{}_{}_preds.jsonl".format(
        opt.dset_name, opt.eval_split_name)

    for epoch_i in trange(start_epoch, opt.n_epoch, desc="Epoch"):
        if epoch_i > -1:
            losses, iteration = train_epoch(model, criterion, train_loader,
                                            optimizer, opt, epoch_i, tb_writer)
            lr_scheduler.step(losses)
        eval_epoch_interval = opt.eval_epoch

        if opt.eval_path is not None and (epoch_i +
                                          1) % eval_epoch_interval == 0:
            with torch.no_grad():
                metrics_no_nms, metrics_nms, eval_loss_meters, latest_file_paths = (
                    eval_epoch(
                        model,
                        val_dataset,
                        opt,
                        save_submission_filename,
                        epoch_i,
                        criterion,
                        tb_writer,
                    ))

            # log
            to_write = opt.eval_log_txt_formatter.format(
                time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
                epoch=epoch_i,
                loss_str=" ".join([
                    "{} {:.4f}".format(k, v.avg)
                    for k, v in eval_loss_meters.items()
                ]),
                eval_metrics_str=json.dumps(metrics_no_nms),
            )

            with open(opt.eval_log_filepath, "a") as f:
                f.write(to_write)
            logger.info("metrics_no_nms {}".format(
                pprint.pformat(metrics_no_nms["brief"], indent=4)))
            if metrics_nms is not None:
                logger.info("metrics_nms {}".format(
                    pprint.pformat(metrics_nms["brief"], indent=4)))

            metrics = metrics_no_nms
            for k, v in metrics["brief"].items():
                tb_writer.add_scalar(f"Eval/{k}", float(v), iteration)

            if opt.dset_name in ["hl"]:
                stop_score = metrics["brief"]["MR-full-mAP"]
            elif opt.dset_name in ["tacos"]:
                stop_score = metrics["brief"]["MR-full-R1@0.3"]
            else:
                stop_score = (metrics["brief"]["MR-full-R1@0.7"] +
                              metrics["brief"]["MR-full-R1@0.5"]) / 2

            if stop_score > prev_best_score:
                es_cnt = 0
                prev_best_score = stop_score

                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch_i,
                    "opt": opt,
                }
                torch.save(checkpoint,
                           opt.ckpt_filepath.replace(".ckpt", "_best.ckpt"))

                best_file_paths = [
                    e.replace("latest", "best") for e in latest_file_paths
                ]
                for src, tgt in zip(latest_file_paths, best_file_paths):
                    os.renames(src, tgt)
                logger.info("The checkpoint file has been updated.")
            else:
                es_cnt += 1
                if opt.max_es_cnt != -1 and es_cnt > opt.max_es_cnt:  # early stop
                    with open(opt.train_log_filepath, "a") as f:
                        f.write(f"Early Stop at epoch {epoch_i}")
                    logger.info(
                        f"\n>>>>> Early stop at epoch {epoch_i}  {prev_best_score}\n"
                    )
                    break

        # save ckpt
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch_i,
            "opt": opt,
        }
        torch.save(checkpoint,
                   opt.ckpt_filepath.replace(".ckpt", "_latest.ckpt"))

        if opt.debug:
            break

    tb_writer.close()


def train_hl(model, criterion, optimizer, lr_scheduler, train_dataset,
             val_dataset, opt):
    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)

    tb_writer = SummaryWriter(opt.tensorboard_log_dir)
    tb_writer.add_text("hyperparameters",
                       dict_to_markdown(vars(opt), max_str_len=None))
    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"

    train_loader = DataLoader(
        train_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.bsz,
        num_workers=opt.num_workers,
        shuffle=True,
        pin_memory=opt.pin_memory,
        drop_last=opt.drop_last,
    )

    prev_best_score = 0.0
    es_cnt = 0
    # start_epoch = 0
    if opt.start_epoch is None:
        start_epoch = -1 if opt.eval_untrained else 0
    else:
        start_epoch = opt.start_epoch
    save_submission_filename = "latest_{}_{}_preds.jsonl".format(
        opt.dset_name, opt.eval_split_name)
    for epoch_i in trange(start_epoch, opt.n_epoch, desc="Epoch"):
        if epoch_i > -1:
            train_epoch(model, criterion, train_loader, optimizer, opt,
                        epoch_i, tb_writer)
            lr_scheduler.step()  # use step() for StepLR not ReduceLROnPlateau
        eval_epoch_interval = opt.eval_epoch
        if opt.eval_path is not None and (epoch_i +
                                          1) % eval_epoch_interval == 0:
            with torch.no_grad():
                metrics_no_nms, metrics_nms, eval_loss_meters, latest_file_paths = (
                    eval_epoch(
                        model,
                        val_dataset,
                        opt,
                        save_submission_filename,
                        epoch_i,
                        criterion,
                        tb_writer,
                    ))

            # log
            to_write = opt.eval_log_txt_formatter.format(
                time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
                epoch=epoch_i,
                loss_str=" ".join([
                    "{} {:.4f}".format(k, v.avg)
                    for k, v in eval_loss_meters.items()
                ]),
                eval_metrics_str=json.dumps(metrics_no_nms),
            )

            with open(opt.eval_log_filepath, "a") as f:
                f.write(to_write)
            logger.info("metrics_no_nms {}".format(
                pprint.pformat(metrics_no_nms["brief"], indent=4)))
            if metrics_nms is not None:
                logger.info("metrics_nms {}".format(
                    pprint.pformat(metrics_nms["brief"], indent=4)))

            metrics = metrics_no_nms
            for k, v in metrics["brief"].items():
                tb_writer.add_scalar(f"Eval/{k}", float(v), epoch_i + 1)

            stop_score = metrics["brief"]["mAP"]
            if stop_score > prev_best_score:
                es_cnt = 0
                prev_best_score = stop_score

                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch_i,
                    "opt": opt,
                }
                torch.save(checkpoint,
                           opt.ckpt_filepath.replace(".ckpt", "_best.ckpt"))

                best_file_paths = [
                    e.replace("latest", "best") for e in latest_file_paths
                ]
                for src, tgt in zip(latest_file_paths, best_file_paths):
                    os.renames(src, tgt)
                logger.info("The checkpoint file has been updated.")
            else:
                es_cnt += 1
                if opt.max_es_cnt != -1 and es_cnt > opt.max_es_cnt:  # early stop
                    with open(opt.train_log_filepath, "a") as f:
                        f.write(f"Early Stop at epoch {epoch_i}")
                    logger.info(
                        f"\n>>>>> Early stop at epoch {epoch_i}  {prev_best_score}\n"
                    )
                    break

            # save ckpt
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch_i,
                "opt": opt,
            }
            torch.save(checkpoint,
                       opt.ckpt_filepath.replace(".ckpt", "_latest.ckpt"))

        if opt.debug:
            break

    tb_writer.close()


def train_hl_with_gan(model, criterion, optimizers, lr_scheduler,
                      train_dataset, val_dataset, opt):
    optimizer_g, optimizer_d = optimizers

    if opt.device.type == "cuda":
        logger.info("CUDA enabled for GAN training")
        model.to(opt.device)

    tb_writer = SummaryWriter(opt.tensorboard_log_dir)
    tb_writer.add_text("hyperparameters",
                       dict_to_markdown(vars(opt), max_str_len=None))
    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"

    train_loader = DataLoader(
        train_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.bsz,
        num_workers=opt.num_workers,
        shuffle=True,
        pin_memory=opt.pin_memory,
        drop_last=opt.drop_last,
    )

    prev_best_score = 0.0
    es_cnt = 0
    start_epoch = opt.start_epoch if opt.start_epoch is not None else 0
    save_submission_filename = f"latest_{opt.dset_name}_{opt.eval_split_name}_preds.jsonl"

    for epoch_i in trange(start_epoch, opt.n_epoch, desc="Epoch"):
        model.train()
        loss_meters = defaultdict(AverageMeter)

        for batch_idx, batch in tqdm(enumerate(train_loader),
                                     desc="Training Iteration"):
            # Prepare inputs
            model_inputs, targets = prepare_batch_inputs(
                batch[1], opt.device, non_blocking=opt.pin_memory)
            targets["label"] = batch[0]
            vid_feats = model_inputs["src_vid"]
            max_v_len = vid_feats.size(1)

            # Process labels with padding
            gt_highlight = []
            for item in batch[0]:
                label = torch.tensor(item["label"]).float().flatten()
                if len(label) > max_v_len:
                    label = label[:max_v_len]  # Truncate if too long
                else:
                    pad_size = max_v_len - len(label)
                    label = torch.cat([label, torch.zeros(pad_size)])
                gt_highlight.append(label)
            gt_highlight = torch.stack(gt_highlight).to(opt.device)

            # ===== 1. Train Discriminator =====
            with torch.no_grad():
                outputs = model(**model_inputs, targets=targets)
                pred_scores = outputs["saliency_scores"]
                mask = (pred_scores > 0.5).float().unsqueeze(-1)
                fake_features = vid_feats * mask

            real_mask = gt_highlight.unsqueeze(-1)
            real_features = vid_feats * real_mask

            # Discriminator forward
            # print("Real features shape:", real_features.shape)
            # print("Fake features shape:", fake_features.shape)
            d_real = model.discriminator(real_features)
            d_fake = model.discriminator(fake_features)

            # Discriminator loss
            if opt.gan_dis_loss == 1:
                d_loss, loss_dict = wgan_gp_mse_loss(
                    d_real=d_real,
                    d_fake=d_fake,
                    real_feats=real_features,
                    fake_feats=fake_features,
                    discriminator=model.discriminator,
                    lambda_gp=10,  # From WGAN-GP paper
                    lambda_mse=0.1  # Tune based on your task
                )
            elif opt.gan_dis_loss == 2:
                d_loss, _ = feature_matching_loss(d_real, d_fake)

            # Discriminator update
            optimizer_d.zero_grad()
            d_loss.backward()
            if opt.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.discriminator.parameters(),
                                         opt.grad_clip)
            optimizer_d.step()

            # ===== 2. Train Generator =====
            outputs = model(**model_inputs, targets=targets)
            pred_scores = outputs["saliency_scores"]

            # Original loss
            original_loss = F.binary_cross_entropy_with_logits(
                pred_scores, gt_highlight)

            # Adversarial loss
            current_mask = (pred_scores > 0.5).float().unsqueeze(-1)
            fake_features = vid_feats * current_mask
            d_fake = model.discriminator(fake_features)
            adversarial_loss = -torch.log(d_fake + 1e-8).mean()

            # Total loss
            total_loss = original_loss + opt.lambda_gan * adversarial_loss

            # Generator update
            optimizer_g.zero_grad()
            total_loss.backward()
            if opt.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.flash_vtg.parameters(),
                                         opt.grad_clip)
            optimizer_g.step()

            # Logging
            loss_meters["loss_main"].update(original_loss.item())
            loss_meters["loss_gan"].update(adversarial_loss.item())
            loss_meters["loss_disc"].update(d_loss.item())

        # ===== Evaluation =====
        if (epoch_i + 1) % opt.eval_epoch == 0:
            with torch.no_grad():
                # Evaluate using original model without discriminator
                metrics_no_nms, metrics_nms, eval_loss_meters, _ = eval_epoch(
                    model.flash_vtg,
                    val_dataset,
                    opt,
                    save_submission_filename,
                    epoch_i,
                    criterion,
                    tb_writer,
                )

            # Log metrics
            current_metrics = metrics_no_nms["brief"]
            logger.info(
                f"Epoch {epoch_i+1} metrics: {pprint.pformat(current_metrics, indent=4)}"
            )

            # Checkpoint saving
            checkpoint = {
                "model": model.flash_vtg.state_dict(),
                "discriminator": model.discriminator.state_dict(),
                "optimizer_g": optimizer_g.state_dict(),
                "optimizer_d": optimizer_d.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch_i,
                "opt": opt,
            }
            torch.save(checkpoint,
                       opt.ckpt_filepath.replace(".ckpt", "_latest.ckpt"))

            # Early stopping logic
            stop_score = current_metrics["mAP"]
            if stop_score > prev_best_score:
                # Save best checkpoint
                torch.save(checkpoint,
                           opt.ckpt_filepath.replace(".ckpt", "_best.ckpt"))
                prev_best_score = stop_score
                es_cnt = 0
            else:
                es_cnt += 1
                if opt.max_es_cnt != -1 and es_cnt > opt.max_es_cnt:
                    logger.info(f"Early stopping at epoch {epoch_i}")
                    break

        # Epoch logging
        tb_writer.add_scalar("Train/loss_main", loss_meters["loss_main"].avg,
                             epoch_i)
        tb_writer.add_scalar("Train/loss_gan", loss_meters["loss_gan"].avg,
                             epoch_i)
        tb_writer.add_scalar("Train/loss_disc", loss_meters["loss_disc"].avg,
                             epoch_i)
        tb_writer.add_scalar("LR", optimizer_g.param_groups[0]["lr"], epoch_i)

    tb_writer.close()
    return opt.ckpt_filepath.replace(".ckpt", "_best.ckpt")


def train_with_gan(model, criterion, optimizers, lr_scheduler, train_dataset,
                   val_dataset, opt):
    """
    GAN training loop for Moment Retrieval (MR) tasks.
    Includes full logging and stop_score logic matching the original train function.
    """
    optimizer_g, optimizer_d = optimizers

    if opt.device.type == "cuda":
        logger.info("CUDA enabled for GAN training (MR)")
        model.to(opt.device)

    tb_writer = SummaryWriter(opt.tensorboard_log_dir)
    tb_writer.add_text("hyperparameters",
                       dict_to_markdown(vars(opt), max_str_len=None))
    
    # Formatters
    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"

    train_loader = DataLoader(
        train_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.bsz,
        num_workers=opt.num_workers,
        shuffle=True,
        pin_memory=opt.pin_memory,
    )

    prev_best_score = 0.0
    es_cnt = 0
    start_epoch = opt.start_epoch if opt.start_epoch is not None else 0
    save_submission_filename = f"latest_{opt.dset_name}_{opt.eval_split_name}_preds.jsonl"

    # Default GAN weight if not specified
    lambda_gan = getattr(opt, 'lambda_gan', 0.1)

    for epoch_i in trange(start_epoch, opt.n_epoch, desc="Epoch"):
        model.train()
        loss_meters = defaultdict(AverageMeter)

        for batch_idx, batch in tqdm(enumerate(train_loader),
                                     desc="Training Iteration",
                                     total=len(train_loader)):
            
            # --- Training Steps (Same as before) ---
            # 1. Prepare Data
            model_inputs, targets = prepare_batch_inputs(
                batch[1], opt.device, non_blocking=opt.pin_memory)
            targets["label"] = batch[0]
            targets["fps"] = torch.full((len(batch[0]), ), 1 / opt.clip_length).to(opt.device)
            
            vid_feats = model_inputs["src_vid"] # (B, T, D)
            B, T, D = vid_feats.shape

            # 2. Construct Real Mask from MR Windows
            # real_mask = torch.zeros((B, T), device=opt.device)
            # fps = 1 / opt.clip_length
            
            # for i, label_item in enumerate(batch[0]):
            #     windows = []
            #     if 'relevant_windows' in label_item:
            #         windows = label_item['relevant_windows'] 
            #     elif 'windows' in label_item: 
            #         windows = label_item['windows']

            #     for start, end in windows:
            #         s_idx = min(int(start * fps), T-1)
            #         e_idx = min(int(end * fps), T-1)
            #         if s_idx <= e_idx:
            #             real_mask[i, s_idx:e_idx+1] = 1.0
            
            # real_mask = real_mask.unsqueeze(-1) 
            # real_features = vid_feats * real_mask

            # 2. Construct Real Mask from MR Windows (Gaussian Fix)
            real_mask = torch.zeros((B, T), device=opt.device)
            
            for i, label_item in enumerate(batch[0]):
                windows = []
                if 'relevant_windows' in label_item:
                    windows = label_item['relevant_windows'] 
                elif 'windows' in label_item: 
                    windows = label_item['windows']
                
                if len(windows) == 0: continue

                duration = label_item.get('duration', T * opt.clip_length)
                c_list = []
                w_list = []
                
                for start, end in windows:
                    start = max(0, min(start, duration))
                    end = max(0, min(end, duration))
                    
                    if end <= start: continue

                    center_sec = (start + end) / 2.0
                    width_sec = end - start
                    
                    # Normalize to [0, 1]
                    c_norm = center_sec / duration
                    w_norm = width_sec / duration
                    
                    c_list.append(c_norm)
                    w_list.append(w_norm)
                
                if len(c_list) > 0:
                    c_tensor = torch.tensor(c_list, device=opt.device)
                    w_tensor = torch.tensor(w_list, device=opt.device)
                    
                    # Generate Gaussian for each window
                    g_masks = gaussian_mask_1d(c_tensor, w_tensor, T, opt.device)
                    
                    # Max-pool to combine multiple windows
                    batch_mask, _ = torch.max(g_masks, dim=0)
                    real_mask[i] = batch_mask

            real_mask = real_mask.unsqueeze(-1) # (B, T, 1)
            real_features = vid_feats * real_mask

            # Phase A: Train Discriminator
            optimizer_d.zero_grad()
            with torch.no_grad():
                outputs = model(**model_inputs, targets=targets)
                fake_features = outputs["fake_features"].detach() 

            txt_pooled = outputs["txt_pooled"].detach()
            d_real = model.discriminator(real_features, txt_emb=txt_pooled)
            d_fake = model.discriminator(fake_features.detach(), txt_emb=txt_pooled)

            d_loss, _ = wgan_gp_mse_loss(
                d_real, d_fake, real_features, fake_features, 
                model.discriminator, txt_emb=txt_pooled
            )
            d_loss.backward()
            optimizer_d.step()
            loss_meters["loss_d"].update(d_loss.item())

            # Phase B: Train Generator
            optimizer_g.zero_grad()
            outputs = model(**model_inputs, targets=targets)
            fake_features = outputs["fake_features"]

            loss_dict = criterion(batch, outputs, targets)
            loss_dict_filtered = {k: v for k, v in outputs.items() if 'loss' in k}
            weight_dict = criterion.weight_dict
            task_loss = sum(loss_dict_filtered[k] * weight_dict[k] 
                           for k in loss_dict_filtered.keys() if k in weight_dict)

            d_fake_pred = model.discriminator(fake_features, txt_emb=txt_pooled)
            adv_loss = -torch.mean(d_fake_pred)
            # total_g_loss = task_loss + lambda_gan * adv_loss
            # Warmup: Don't use GAN loss for first 5 epochs
            current_lambda = 0.0 if epoch_i < 5 else lambda_gan
            total_g_loss = task_loss + current_lambda * adv_loss

            total_g_loss.backward()
            if opt.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.flash_vtg.parameters(), opt.grad_clip)
            optimizer_g.step()

            # Logging
            loss_meters["loss_task"].update(task_loss.item())
            loss_meters["loss_adv"].update(adv_loss.item())
            for k, v in loss_dict_filtered.items():
                loss_meters[k].update(float(v))
            
            current_iter = epoch_i * len(train_loader) + batch_idx
            tb_writer.add_scalar("Train/loss_d", d_loss.item(), current_iter)
            tb_writer.add_scalar("Train/loss_task", task_loss.item(), current_iter)
            tb_writer.add_scalar("Train/loss_adv", adv_loss.item(), current_iter)

        # ===============================================================
        #  Evaluation (Restored Full Logging Logic)
        # ===============================================================
        if (epoch_i + 1) % opt.eval_epoch == 0:
            with torch.no_grad():
                metrics_no_nms, metrics_nms, eval_loss_meters, latest_file_paths = (
                    eval_epoch(
                        model.flash_vtg, # Eval only the inner model
                        val_dataset,
                        opt,
                        save_submission_filename,
                        epoch_i,
                        criterion,
                        tb_writer,
                    ))

            # --- Full Logging Block from Original Code ---
            to_write = opt.eval_log_txt_formatter.format(
                time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
                epoch=epoch_i,
                loss_str=" ".join([
                    "{} {:.4f}".format(k, v.avg)
                    for k, v in eval_loss_meters.items()
                ]),
                eval_metrics_str=json.dumps(metrics_no_nms),
            )

            with open(opt.eval_log_filepath, "a") as f:
                f.write(to_write)
            
            logger.info("metrics_no_nms {}".format(
                pprint.pformat(metrics_no_nms["brief"], indent=4)))
            
            if metrics_nms is not None:
                logger.info("metrics_nms {}".format(
                    pprint.pformat(metrics_nms["brief"], indent=4)))

            metrics = metrics_no_nms
            # Calculate current iteration for TB logging
            iteration = (epoch_i + 1) * len(train_loader)
            
            for k, v in metrics["brief"].items():
                tb_writer.add_scalar(f"Eval/{k}", float(v), iteration)

            # --- Stop Score Logic from Original Code ---
            if opt.dset_name in ["hl"]:
                stop_score = metrics["brief"]["MR-full-mAP"]
            elif opt.dset_name in ["tacos"]:
                stop_score = metrics["brief"]["MR-full-R1@0.3"]
            else:
                # Default logic for Charades/QVHighlights/etc
                stop_score = (metrics["brief"]["MR-full-R1@0.7"] +
                              metrics["brief"]["MR-full-R1@0.5"]) / 2

            # Checkpoint Logic
            if stop_score > prev_best_score:
                es_cnt = 0
                prev_best_score = stop_score
                
                checkpoint = {
                    "model": model.flash_vtg.state_dict(),
                    "discriminator": model.discriminator.state_dict(),
                    "optimizer_g": optimizer_g.state_dict(),
                    "optimizer_d": optimizer_d.state_dict(),
                    "epoch": epoch_i,
                    "opt": opt,
                }
                torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", "_best.ckpt"))
                
                # Rename prediction files
                best_file_paths = [e.replace("latest", "best") for e in latest_file_paths]
                for src, tgt in zip(latest_file_paths, best_file_paths):
                    if os.path.exists(src):
                        os.renames(src, tgt)
                        
                logger.info("The checkpoint file has been updated.")
            else:
                es_cnt += 1
                if opt.max_es_cnt != -1 and es_cnt > opt.max_es_cnt:
                    with open(opt.train_log_filepath, "a") as f:
                        f.write(f"Early Stop at epoch {epoch_i}")
                    logger.info(f"\n>>>>> Early stop at epoch {epoch_i}  {prev_best_score}\n")
                    break

            # Save Latest Checkpoint
            checkpoint = {
                "model": model.flash_vtg.state_dict(),
                "discriminator": model.discriminator.state_dict(),
                "optimizer_g": optimizer_g.state_dict(),
                "optimizer_d": optimizer_d.state_dict(),
                "epoch": epoch_i,
                "opt": opt,
            }
            torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", "_latest.ckpt"))

    tb_writer.close()
    return opt.ckpt_filepath.replace(".ckpt", "_best.ckpt")


def start_training():
    logger.info("Setup data and model...")

    dataset_config = dict(
        dset_name=opt.dset_name,
        data_path=opt.train_path,
        v_feat_dirs=opt.v_feat_dirs,
        q_feat_dir=opt.t_feat_dir,
        q_feat_type=opt.q_feat_type,
        max_q_l=opt.max_q_l,
        max_v_l=opt.max_v_l,
        ctx_mode=opt.ctx_mode,
        data_ratio=opt.data_ratio,
        normalize_v=not opt.no_norm_vfeat,
        normalize_t=not opt.no_norm_tfeat,
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        span_loss_type=opt.span_loss_type,
        txt_drop_ratio=opt.txt_drop_ratio,
        dset_domain=opt.dset_domain,
        add_gan=opt.add_gan,
    )
    dataset_config["data_path"] = opt.train_path
    train_dataset = StartEndDataset(**dataset_config)

    if opt.eval_path is not None:
        dataset_config["data_path"] = opt.eval_path
        dataset_config["txt_drop_ratio"] = 0
        dataset_config["q_feat_dir"] = opt.t_feat_dir.replace(
            "sub_features", "text_features")  # for pretraining
        # dataset_config["load_labels"] = False  # uncomment to calculate eval loss

        eval_dataset = StartEndDataset(**dataset_config)

    else:
        eval_dataset = None

    if not opt.add_gan:
        model, criterion, optimizer, lr_scheduler = setup_model(opt)
    else:
        model, criterion, optimizers, lr_scheduler = setup_model_with_gan(opt)
    logger.info(f"Model {model}")
    params = []
    logger.info("Learnable Parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            # logger.info(f"{name} - {param.shape}")
            params.append(param)

    train_params = sum(p.numel() for p in params)
    total_params = sum(p.numel() for p in model.parameters())
    ratio = round(train_params / total_params * 100, 3)
    param = round(train_params / 1024 / 1024, 3)
    logger.info(f"Learnable Parameters: {param}M ({ratio}%)")

    logger.info("Start Training...")

    # For tvsum dataset, use train_hl function
    if opt.dset_name in ['tvsum', 'youtube_uni']:
        if not opt.add_gan:
            train_hl(model, criterion, optimizer, lr_scheduler, train_dataset,
                     eval_dataset, opt)
        else:
            train_hl_with_gan(model, criterion, optimizers, lr_scheduler,
                              train_dataset, eval_dataset, opt)
    else:
        # MR Tasks (QVHighlights, Charades, etc.)
        if not opt.add_gan:
            train(model, criterion, optimizer, lr_scheduler, train_dataset,
                  eval_dataset, opt)
        else:
            # === NEW CALL ===
            train_with_gan(model, criterion, optimizers, lr_scheduler, 
                           train_dataset, eval_dataset, opt)
        # train(model, criterion, optimizer, lr_scheduler, train_dataset,
        #       eval_dataset, opt)
    return (
        opt.ckpt_filepath.replace(".ckpt", "_best.ckpt"),
        opt.eval_split_name,
        opt.eval_path,
        opt.debug,
        opt,
    )


if __name__ == "__main__":
    opt = BaseOptions().parse()
    set_seed(opt.seed)
    if opt.debug:  # keep the model run deterministically
        # 'cudnn.benchmark = True' enabled auto finding the best algorithm for a specific input/net config.
        # Enable this only when input size is fixed.
        cudnn.benchmark = False
        cudnn.deterministic = True

    opt.cfg = nncore.Config.from_file(opt.config)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    log_directory = os.path.join(
        opt.results_dir,
        datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.log')
    file_handler = logging.FileHandler(log_directory)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    best_ckpt_path, eval_split_name, eval_path, debug, opt = start_training()

    if not debug:
        input_args = [
            opt.config,
            "--resume",
            best_ckpt_path,
            "--eval_split_name",
            eval_split_name,
            "--eval_path",
            eval_path,
        ]

        import sys

        sys.argv[1:] = input_args
        logger.info("\n\n\nFINISHED TRAINING!!!")
        logger.info("Evaluating model at {}".format(best_ckpt_path))
        logger.info("Input args {}".format(sys.argv[1:]))
        start_inference(opt)
