"""
Production-Grade Virtual Try-On with Spatially-Adaptive GANs (SPADE) - SELF-CONTAINED
=====================================================================================
This script implements a complete, end-to-end training system for virtual
try-on using a conditional GAN based on the SPADE architecture. It is
designed to be self-contained, with no external dependencies on other
project scripts.
"""

import os
import sys
import json
import time
import torch
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms as transforms
import argparse

# This script is now self-contained and does not require the problematic environment variable.
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True' 
warnings.filterwarnings("ignore")

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- DATASET CLASSES (Copied from kaggle_training.py to remove dependency) ---

@dataclass
class DatasetConfig:
    """Configuration class specifically for the EnhancedOOTDDataset."""
    # This is a subset of the original TrainingConfig, containing only what the dataset needs.
    data_root: str = "./dataset"
    image_size: int = 256
    normalize_images: bool = True
    image_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    validate_data_loading: bool = False # Keep this false for GAN training to avoid extra overhead
    save_debug_samples: bool = False
    max_debug_samples: int = 5
    output_dir: str = "./gan_tryon_output"
    debug_dir: str = "debug_samples"

class EnhancedOOTDDataset(Dataset):
    """Enhanced dataset with proper data validation and debugging."""
    def __init__(self, data_root: str, config: DatasetConfig):
        self.data_root = Path(data_root)
        self.config = config
        self.image_size = config.image_size
        self.validation_stats = {
            'total_pairs': 0, 'valid_pairs': 0, 'missing_files': [], 'corrupted_files': [],
            'keypoint_stats': {'empty': 0, 'valid': 0, 'total_keypoints': 0}
        }
        
        if config.normalize_images:
            self.image_normalize = transforms.Normalize(mean=list(config.image_mean), std=list(config.image_std))
            self.denormalize = transforms.Normalize(
                mean=[-m/s for m, s in zip(config.image_mean, config.image_std)],
                std=[1/s for s in config.image_std]
            )
        else:
            self.image_normalize = None
            self.denormalize = None
        
        self._load_and_validate_dataset()
        logger.info(f"Dataset: {len(self.pairs)} samples loaded.")

    def _load_and_validate_dataset(self):
        pairs_dir = self.data_root / "pairs"
        if not pairs_dir.exists():
            raise FileNotFoundError(f"Pairs directory not found at {pairs_dir}")
        
        all_pairs = [p.name for p in pairs_dir.iterdir() if p.is_dir()]
        self.validation_stats['total_pairs'] = len(all_pairs)
        
        valid_pairs = []
        for pair_id in all_pairs:
            if self._validate_pair(pair_id):
                valid_pairs.append(pair_id)
        
        self.pairs = valid_pairs
        self.validation_stats['valid_pairs'] = len(valid_pairs)

    def _validate_pair(self, pair_id: str) -> bool:
        pair_dir = self.data_root / "pairs" / pair_id
        required_files = ["human.png", "cloth.png", "cloth_mask.png"]
        for filename in required_files:
            if not (pair_dir / filename).exists():
                return False
        return True

    def __len__(self):
        return len(self.pairs)

    def _load_image(self, path: Path, normalize: bool = True) -> torch.Tensor:
        try:
            image = Image.open(path).convert('RGB').resize((self.image_size, self.image_size), Image.LANCZOS)
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)
            if normalize and self.image_normalize is not None:
                image = self.image_normalize(image)
            else:
                image = image * 2.0 - 1.0
            return image
        except Exception as e:
            logger.error(f"Error loading image {path}: {e}")
            return torch.zeros(3, self.image_size, self.image_size)

    def _load_mask(self, path: Path) -> torch.Tensor:
        try:
            mask = Image.open(path).convert('L').resize((self.image_size, self.image_size), Image.NEAREST)
            mask = np.array(mask).astype(np.float32) / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0)
            return mask * 2.0 - 1.0
        except Exception as e:
            logger.error(f"Error loading mask {path}: {e}")
            return torch.zeros(1, self.image_size, self.image_size)

    def _load_keypoints(self, path: Path) -> torch.Tensor:
        try:
            with open(path, 'r') as f:
                kpts_data = json.load(f)
            
            keypoints = torch.zeros(18, self.image_size, self.image_size)
            pose_keypoints = kpts_data.get('pose_keypoints_2d', [])
            
            for i, kpt in enumerate(pose_keypoints[:18]):
                if len(kpt) >= 2:
                    x, y = float(kpt[0]), float(kpt[1])
                    x_scaled = int(x * self.image_size)
                    y_scaled = int(y * self.image_size)
                    if 0 <= x_scaled < self.image_size and 0 <= y_scaled < self.image_size:
                        keypoints[i, y_scaled, x_scaled] = 1.0
            return keypoints * 2.0 - 1.0
        except Exception:
            return torch.zeros(18, self.image_size, self.image_size)

    def _load_parsing(self, path: Path) -> torch.Tensor:
        try:
            parsing = Image.open(path).convert('L').resize((self.image_size, self.image_size), Image.NEAREST)
            parsing = np.array(parsing).astype(np.float32) / 255.0
            parsing = torch.from_numpy(parsing).unsqueeze(0)
            return parsing * 2.0 - 1.0
        except Exception:
            return torch.zeros(1, self.image_size, self.image_size)

    def __getitem__(self, idx):
        pair_id = self.pairs[idx]
        pair_dir = self.data_root / "pairs" / pair_id
        
        human_image = self._load_image(pair_dir / "human.png")
        cloth_image = self._load_image(pair_dir / "cloth.png")
        cloth_mask = self._load_mask(pair_dir / "cloth_mask.png")
        
        keypoints = self._load_keypoints(pair_dir / "keypoints.json") if (pair_dir / "keypoints.json").exists() else torch.zeros(18, self.image_size, self.image_size)
        parsing = self._load_parsing(pair_dir / "parsing.png") if (pair_dir / "parsing.png").exists() else torch.zeros(1, self.image_size, self.image_size)
        
        return {
            'human_image': human_image, 'cloth_image': cloth_image, 'cloth_mask': cloth_mask,
            'keypoints': keypoints, 'parsing': parsing
        }

# --- GAN Configuration ---
@dataclass
class GANTrainingConfig:
    """Configuration for SPADE GAN-based Virtual Try-On Training"""
    data_root: str = "./dataset"
    output_dir: str = "./gan_tryon_output"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 2
    image_size: int = 256
    semantic_channels: int = 19 # 18 keypoints + 1 parsing map
    cloth_embedding_dim: int = 256
    batch_size: int = 4
    num_epochs: int = 300
    lr_g: float = 1e-4
    lr_d: float = 4e-4
    beta1: float = 0.0
    beta2: float = 0.999
    lambda_feat: float = 10.0
    lambda_vgg: float = 10.0
    lambda_l1: float = 5.0
    log_every: int = 20
    validate_every: int = 1
    save_every: int = 1
    max_validation_samples: int = 4
    
    def __post_init__(self):
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        self.results_dir = os.path.join(self.output_dir, "results")
        self.logs_dir = os.path.join(self.output_dir, "logs")
        for path in [self.output_dir, self.checkpoint_dir, self.results_dir, self.logs_dir]:
            os.makedirs(path, exist_ok=True)

# --- GAN Loss Functions ---
class GANLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def __call__(self, predictions, is_real):
        if is_real:
            return F.relu(1.0 - predictions).mean()
        else:
            return F.relu(1.0 + predictions).mean() if is_real is False else -predictions.mean()

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features
        self.slices = nn.ModuleList([vgg[:2], vgg[2:7], vgg[7:12], vgg[12:21], vgg[21:30]])
        for param in self.parameters():
            param.requires_grad = False
    def forward(self, x, y):
        loss = 0.0
        for slice in self.slices:
            x, y = slice(x), slice(y)
            loss += F.l1_loss(x, y)
        return loss

# --- SPADE Architecture ---
class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()
        self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        nhidden = 128
        self.mlp_shared = nn.Sequential(nn.Conv2d(label_nc, nhidden, 3, 1, 1), nn.ReLU())
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, 3, 1, 1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, 3, 1, 1)
    def forward(self, x, segmap):
        normalized_input = self.param_free_norm(x)
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        return normalized_input * (1 + gamma) + beta

class SPADEResBlk(nn.Module):
    def __init__(self, fin, fout, label_nc):
        super().__init__()
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        self.conv_0 = nn.Conv2d(fin, fmiddle, 3, 1, 1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, 3, 1, 1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, 1, bias=False)
        self.norm_0 = SPADE(fin, label_nc)
        self.norm_1 = SPADE(fmiddle, label_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(fin, label_nc)
    def forward(self, x, segmap):
        x_s = self.shortcut(x, segmap)
        dx = self.conv_0(F.leaky_relu(self.norm_0(x, segmap), 0.2))
        dx = self.conv_1(F.leaky_relu(self.norm_1(dx, segmap), 0.2))
        return x_s + dx
    def shortcut(self, x, segmap):
        return self.conv_s(self.norm_s(x, segmap)) if self.learned_shortcut else x

# --- Generator ---
class SPADEGenerator(nn.Module):
    def __init__(self, config: GANTrainingConfig):
        super().__init__()
        self.config = config
        nf = 64
        self.cloth_encoder = nn.Sequential(
            nn.Conv2d(4, 32, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, config.cloth_embedding_dim, 1)
        )
        self.fc = nn.Conv2d(config.semantic_channels, 16 * nf, 3, 1, 1)
        self.head_0 = SPADEResBlk(16 * nf, 16 * nf, config.semantic_channels + config.cloth_embedding_dim)
        self.g_0 = SPADEResBlk(16 * nf, 16 * nf, config.semantic_channels + config.cloth_embedding_dim)
        self.g_1 = SPADEResBlk(16 * nf, 8 * nf, config.semantic_channels + config.cloth_embedding_dim)
        self.g_2 = SPADEResBlk(8 * nf, 4 * nf, config.semantic_channels + config.cloth_embedding_dim)
        self.g_3 = SPADEResBlk(4 * nf, 2 * nf, config.semantic_channels + config.cloth_embedding_dim)
        self.g_4 = SPADEResBlk(2 * nf, 1 * nf, config.semantic_channels + config.cloth_embedding_dim)
        self.conv_img = nn.Conv2d(nf, 3, 3, 1, 1)
        self.up = nn.Upsample(scale_factor=2)
    def forward(self, semantic_map, cloth_condition):
        cloth_embedding = self.cloth_encoder(cloth_condition)
        seg = F.interpolate(semantic_map, size=(self.config.image_size // 16, self.config.image_size // 16), mode='nearest')
        cloth_emb_map = cloth_embedding.expand(-1, -1, seg.shape[2], seg.shape[3])
        combined_cond = torch.cat([seg, cloth_emb_map], dim=1)
        x = self.fc(seg)
        x = self.head_0(x, combined_cond)
        x = self.up(x); x = self.g_0(x, combined_cond)
        x = self.up(x); x = self.g_1(x, combined_cond)
        x = self.up(x); x = self.g_2(x, combined_cond)
        x = self.up(x); x = self.g_3(x, combined_cond)
        x = self.up(x); x = self.g_4(x, combined_cond)
        x = self.conv_img(F.leaky_relu(x, 0.2))
        return torch.tanh(x)

# --- Discriminator ---
class MultiscaleDiscriminator(nn.Module):
    def __init__(self, config: GANTrainingConfig):
        super().__init__()
        self.discriminators = nn.ModuleList([NLayerDiscriminator(config.semantic_channels + 3) for _ in range(3)])
    def forward(self, image, semantic_map):
        outputs = []
        for i, D in enumerate(self.discriminators):
            if i > 0:
                image = F.avg_pool2d(image, 3, 2, 1)
                semantic_map = F.avg_pool2d(semantic_map, 3, 2, 1)
            outputs.append(D(torch.cat([image, semantic_map], dim=1)))
        return outputs

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc):
        super().__init__()
        nf = 64
        sequence = [[nn.Conv2d(input_nc, nf, 4, 2, 1), nn.LeakyReLU(0.2, True)]]
        for n in range(1, 4):
            nf_prev, nf = nf, min(nf * 2, 512)
            sequence += [[nn.Conv2d(nf_prev, nf, 4, 2, 1), nn.InstanceNorm2d(nf), nn.LeakyReLU(0.2, True)]]
        sequence += [[nn.Conv2d(nf, 1, 4, 1, 1)]]
        self.model = nn.ModuleList([nn.Sequential(*s) for s in sequence])
    def forward(self, input):
        res = [input]
        for layer in self.model:
            res.append(layer(res[-1]))
        return res[1:]

# --- Trainer ---
class GANTrainer:
    def __init__(self, config: GANTrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.setup_logging()
        self.netG = SPADEGenerator(config).to(self.device)
        self.netD = MultiscaleDiscriminator(config).to(self.device)
        self.optimizerG = Adam(self.netG.parameters(), lr=config.lr_g, betas=(config.beta1, config.beta2))
        self.optimizerD = Adam(self.netD.parameters(), lr=config.lr_d, betas=(config.beta1, config.beta2))
        self.criterionGAN = GANLoss().to(self.device)
        self.criterionFeat = nn.L1Loss().to(self.device)
        self.criterionVGG = VGGPerceptualLoss().to(self.device)
        self.criterionL1 = nn.L1Loss().to(self.device)
        self.init_dataset()
        self.start_epoch = 0
        self.load_checkpoint()
        logger.info("GAN Trainer initialized successfully.")

    def setup_logging(self):
        log_file = os.path.join(self.config.logs_dir, f"gan_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    def init_dataset(self):
        dataset_config = DatasetConfig(data_root=self.config.data_root)
        self.dataset = EnhancedOOTDDataset(self.config.data_root, dataset_config)
        self.dataloader = DataLoader(
            self.dataset, batch_size=self.config.batch_size, shuffle=True,
            num_workers=self.config.num_workers, pin_memory=True, drop_last=True
        )
        logger.info(f"Dataset loaded with {len(self.dataset)} samples.")

    def train(self):
        logger.info("Starting GAN training...")
        for epoch in range(self.start_epoch, self.config.num_epochs):
            for i, data in enumerate(self.dataloader):
                for key in data:
                    if isinstance(data[key], torch.Tensor):
                        data[key] = data[key].to(self.device)
                
                semantic_map = torch.cat([data['keypoints'], data['parsing']], dim=1)
                cloth_condition = torch.cat([data['cloth_image'], data['cloth_mask']], dim=1)
                real_image = data['human_image']

                # Generator Update
                self.optimizerG.zero_grad()
                fake_image = self.netG(semantic_map, cloth_condition)
                pred_fake = self.netD(fake_image, semantic_map)
                pred_real = self.netD(real_image, semantic_map)
                loss_g_adv = self.criterionGAN(pred_fake[-1], True)
                loss_g_feat = self.calculate_feature_matching_loss(pred_fake, pred_real) * self.config.lambda_feat
                loss_g_vgg = self.criterionVGG(fake_image, real_image) * self.config.lambda_vgg
                mask = (F.interpolate(data['cloth_mask'], size=self.config.image_size) > 0).float()
                loss_g_l1 = self.criterionL1(fake_image * mask, real_image * mask) * self.config.lambda_l1
                loss_g = loss_g_adv + loss_g_feat + loss_g_vgg + loss_g_l1
                loss_g.backward()
                self.optimizerG.step()

                # Discriminator Update
                self.optimizerD.zero_grad()
                with torch.no_grad():
                    fake_image_detached = self.netG(semantic_map, cloth_condition).detach()
                pred_fake_d = self.netD(fake_image_detached, semantic_map)
                pred_real_d = self.netD(real_image, semantic_map)
                loss_d_fake = self.criterionGAN(pred_fake_d[-1], False)
                loss_d_real = self.criterionGAN(pred_real_d[-1], True)
                loss_d = (loss_d_fake + loss_d_real) * 0.5
                loss_d.backward()
                self.optimizerD.step()

                if i % self.config.log_every == 0:
                    logger.info(f"E[{epoch}]B[{i}]|D:{loss_d.item():.4f}|G:{loss_g.item():.4f}")

            if epoch % self.config.validate_every == 0: self.validate(epoch)
            if epoch % self.config.save_every == 0: self.save_checkpoint(epoch)

    def calculate_feature_matching_loss(self, pred_fake, pred_real):
        loss = 0
        for i in range(len(pred_fake) - 1):
            for j in range(len(pred_fake[i])):
                loss += self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach())
        return loss

    def validate(self, epoch):
        self.netG.eval()
        with torch.no_grad():
            data = next(iter(self.dataloader))
            for key in data:
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].to(self.device)
            fake_image = self.netG(torch.cat([data['keypoints'], data['parsing']], dim=1), torch.cat([data['cloth_image'], data['cloth_mask']], dim=1))
            self.save_visualization(epoch, data['human_image'], fake_image, data['cloth_image'])
        self.netG.train()

    def save_visualization(self, epoch, real, fake, cloth):
        num_samples = min(self.config.max_validation_samples, real.size(0))
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
        for i in range(num_samples):
            real_img = (real[i].cpu().permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1)
            fake_img = (fake[i].cpu().permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1)
            cloth_img = (cloth[i].cpu().permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1)
            ax = axes if num_samples == 1 else axes[i]
            ax[0].imshow(cloth_img); ax[0].set_title("Input Cloth"); ax[0].axis('off')
            ax[1].imshow(fake_img); ax[1].set_title("Generated"); ax[1].axis('off')
            ax[2].imshow(real_img); ax[2].set_title("Ground Truth"); ax[2].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.results_dir, f"epoch_{epoch:03d}.png"))
        plt.close()

    def save_checkpoint(self, epoch):
        state = {
            'epoch': epoch, 'netG_state_dict': self.netG.state_dict(), 'netD_state_dict': self.netD.state_dict(),
            'optimizerG_state_dict': self.optimizerG.state_dict(), 'optimizerD_state_dict': self.optimizerD.state_dict()
        }
        torch.save(state, os.path.join(self.config.checkpoint_dir, f'ckpt_epoch_{epoch:03d}.pth'))
        logger.info(f"Saved checkpoint for epoch {epoch}")

    def load_checkpoint(self):
        if not os.path.exists(self.config.checkpoint_dir): return
        checkpoints = sorted([f for f in os.listdir(self.config.checkpoint_dir) if f.endswith('.pth')], reverse=True)
        if not checkpoints: return
        latest = os.path.join(self.config.checkpoint_dir, checkpoints[0])
        logger.info(f"Loading checkpoint from {latest}")
        try:
            state = torch.load(latest, map_location=self.device)
            self.netG.load_state_dict(state['netG_state_dict'])
            self.netD.load_state_dict(state['netD_state_dict'])
            self.optimizerG.load_state_dict(state['optimizerG_state_dict'])
            self.optimizerD.load_state_dict(state['optimizerD_state_dict'])
            self.start_epoch = state['epoch'] + 1
            logger.info(f"Resuming from epoch {self.start_epoch}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}. Starting fresh.")

def main():
    parser = argparse.ArgumentParser(description="Train SPADE GAN for Virtual Try-On")
    parser.add_argument("--data_root", type=str, required=True, help="Path to the dataset root directory.")
    parser.add_argument("--output_dir", type=str, default="./gan_tryon_output", help="Directory for outputs.")
    args = parser.parse_args()
    config = GANTrainingConfig(data_root=args.data_root, output_dir=args.output_dir)
    trainer = GANTrainer(config)
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted. Saving checkpoint.")
        trainer.save_checkpoint(trainer.start_epoch)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
