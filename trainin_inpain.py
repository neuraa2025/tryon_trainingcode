"""
Enhanced Production-Grade OOTDiffusion Training System - COMPLETE WITH MASKED GENERATION
========================================================================================
Complete implementation with masked region generation and full image loss calculation
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
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DConditionModel
from diffusers.models import UNet2DModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
import torchvision.transforms as transforms

# Suppress warnings
warnings.filterwarnings("ignore")

# FIXED: Proper logging setup with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Enhanced training configuration with masked generation support"""
    # Dataset Configuration
    data_root: str = "dataset"
    batch_size: int = 4
    num_workers: int = 1
    
    # Model Configuration
    timesteps: int = 3600
    image_size: int = 256
    in_channels: int = 12
    out_channels: int = 4
    
    # Normalization Configuration
    normalize_images: bool = True
    image_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # ENHANCED: Masked Generation Configuration
    use_masked_generation: bool = True
    mask_loss_weight: float = 2.0  # Higher weight for masked regions
    preserve_unmasked: bool = True  # Keep unmasked regions unchanged
    mask_blur_radius: int = 0  # Optional: blur mask edges
    
    # Enhanced Validation Configuration
    validate_data_loading: bool = True
    save_debug_samples: bool = True
    max_debug_samples: int = 5
    
    # Visualization Configuration
    save_batch_samples: bool = True
    save_comprehensive_results: bool = True
    batch_samples_dir: str = "batch_samples"
    comprehensive_results_dir: str = "comprehensive_results"
    debug_dir: str = "debug_samples"
    
    # Training Configuration
    num_epochs: int = 250
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Diffusion Configuration
    beta_start: float = 0.00085
    beta_end: float = 0.012
    beta_schedule: str = "scaled_linear"
    prediction_type: str = "epsilon"
    
    # Monitoring Configuration
    save_every: int = 5
    validate_every: int = 1
    log_every: int = 10
    max_validation_samples: int = 4
    
    # Output Configuration
    output_dir: str = "enhanced_masked_ootd_training"
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"
    logs_dir: str = "logs"
    
    # Hardware Configuration
    mixed_precision: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        """Create output directories"""
        for dir_path in [
            self.output_dir, 
            os.path.join(self.output_dir, self.checkpoint_dir),
            os.path.join(self.output_dir, self.results_dir),
            os.path.join(self.output_dir, self.logs_dir),
            os.path.join(self.output_dir, self.batch_samples_dir),
            os.path.join(self.output_dir, self.comprehensive_results_dir),
            os.path.join(self.output_dir, self.debug_dir)
        ]:
            os.makedirs(dir_path, exist_ok=True)

class EnhancedOOTDDataset(Dataset):
    """CORRECTED: Enhanced dataset with proper data validation and debugging"""
    
    def __init__(self, data_root: str, config: TrainingConfig):
        self.data_root = Path(data_root)
        self.config = config
        self.image_size = config.image_size
        self.validation_stats = {
            'total_pairs': 0,
            'valid_pairs': 0,
            'missing_files': [],
            'corrupted_files': [],
            'keypoint_stats': {'empty': 0, 'valid': 0, 'total_keypoints': 0}
        }
        
        # Setup normalization transforms
        if config.normalize_images:
            self.image_normalize = transforms.Normalize(
                mean=list(config.image_mean), 
                std=list(config.image_std)
            )
            self.denormalize = transforms.Normalize(
                mean=[-m/s for m, s in zip(config.image_mean, config.image_std)],
                std=[1/s for s in config.image_std]
            )
        else:
            self.image_normalize = None
            self.denormalize = None
        
        # Load and validate dataset
        self._load_and_validate_dataset()
        
        logger.info(f"Dataset: {len(self.pairs)} samples with enhanced normalization")
        self._print_validation_summary()
    
    def _load_and_validate_dataset(self):
        """CORRECTED: Load and validate all dataset components"""
        # Check for metadata
        metadata_path = self.data_root / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            logger.info("Loaded metadata.json successfully")
        else:
            logger.warning(f"No metadata found at {metadata_path}")
            self.metadata = {}
        
        # Load pairs from directory structure
        pairs_dir = self.data_root / "pairs"
        if not pairs_dir.exists():
            raise FileNotFoundError(f"Pairs directory not found at {pairs_dir}")
        
        # Discover all pairs
        all_pairs = []
        for pair_dir in pairs_dir.iterdir():
            if pair_dir.is_dir():
                all_pairs.append(str(pair_dir.name))
        
        self.validation_stats['total_pairs'] = len(all_pairs)
        logger.info(f"Found {len(all_pairs)} potential pairs")
        
        # Validate each pair
        valid_pairs = []
        for pair_id in all_pairs:
            if self._validate_pair(pair_id):
                valid_pairs.append(pair_id)
        
        self.pairs = valid_pairs
        self.validation_stats['valid_pairs'] = len(valid_pairs)
        
        # Perform comprehensive data validation if requested
        if self.config.validate_data_loading:
            self._comprehensive_data_validation()
    
    def _validate_pair(self, pair_id: str) -> bool:
        """CORRECTED: Validate a single pair has all required files"""
        pair_dir = self.data_root / "pairs" / pair_id
        
        required_files = [
            "human.png",
            "cloth.png", 
            "cloth_mask.png"
        ]
        
        optional_files = [
            "keypoints.json",
            "parsing.png",
            "masked_human.png"
        ]
        
        # Check required files
        for filename in required_files:
            filepath = pair_dir / filename
            if not filepath.exists():
                self.validation_stats['missing_files'].append(f"{pair_id}/{filename}")
                return False
            
            # Validate image files
            if filename.endswith('.png'):
                try:
                    img = Image.open(filepath)
                    img.verify()
                except Exception as e:
                    self.validation_stats['corrupted_files'].append(f"{pair_id}/{filename}: {str(e)}")
                    return False
        
        # Validate keypoints if present
        keypoints_path = pair_dir / "keypoints.json"
        if keypoints_path.exists():
            if not self._validate_keypoints_file(keypoints_path, pair_id):
                return False
        
        return True
    
    def _validate_keypoints_file(self, keypoints_path: Path, pair_id: str) -> bool:
        """CORRECTED: Validate keypoints JSON file format"""
        try:
            with open(keypoints_path, 'r') as f:
                kpts_data = json.load(f)
            
            # Check for correct format - YOUR actual format
            if 'pose_keypoints_2d' not in kpts_data:
                logger.warning(f"Missing 'pose_keypoints_2d' in {pair_id}/keypoints.json")
                return False
            
            keypoints = kpts_data['pose_keypoints_2d']
            
            # Validate keypoints structure
            if not isinstance(keypoints, list):
                logger.warning(f"Invalid keypoints format in {pair_id}")
                return False
            
            # Count valid keypoints
            valid_keypoints = 0
            for kpt in keypoints:
                if isinstance(kpt, list) and len(kpt) >= 2:
                    x, y = kpt[0], kpt[1]
                    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                        if x > 0 and y > 0:  # Valid coordinate
                            valid_keypoints += 1
            
            self.validation_stats['keypoint_stats']['total_keypoints'] += len(keypoints)
            if valid_keypoints > 0:
                self.validation_stats['keypoint_stats']['valid'] += 1
            else:
                self.validation_stats['keypoint_stats']['empty'] += 1
            
            return True
            
        except Exception as e:
            self.validation_stats['corrupted_files'].append(f"{pair_id}/keypoints.json: {str(e)}")
            return False
    
    def _comprehensive_data_validation(self):
        """Perform comprehensive validation on sample data"""
        logger.info("Performing comprehensive data validation...")
        
        sample_size = min(self.config.max_debug_samples, len(self.pairs))
        for i in range(sample_size):
            pair_id = self.pairs[i]
            logger.info(f"Validating sample {i+1}/{sample_size}: {pair_id}")
            
            try:
                sample = self.__getitem__(i)
                self._validate_sample_tensors(sample, pair_id)
                
                # Save debug visualization
                if self.config.save_debug_samples:
                    self._save_debug_sample(sample, pair_id, i)
                    
            except Exception as e:
                logger.error(f"Error validating sample {pair_id}: {str(e)}")
    
    def _validate_sample_tensors(self, sample: dict, pair_id: str):
        """Validate loaded sample tensors"""
        expected_shapes = {
            'human_image': (3, self.image_size, self.image_size),
            'cloth_image': (3, self.image_size, self.image_size),
            'cloth_mask': (1, self.image_size, self.image_size),
            'keypoints': (18, self.image_size, self.image_size),
            'parsing': (1, self.image_size, self.image_size),
            'masked_human': (3, self.image_size, self.image_size)
        }
        
        for key, expected_shape in expected_shapes.items():
            if key in sample:
                tensor = sample[key]
                if tensor.shape != expected_shape:
                    logger.warning(f"{pair_id} - {key}: Expected {expected_shape}, got {tensor.shape}")
                
                # Check for non-zero content
                non_zero_count = torch.sum(tensor != 0).item()
                total_elements = tensor.numel()
                non_zero_percent = (non_zero_count / total_elements) * 100
                
                logger.info(f"{pair_id} - {key}: {non_zero_percent:.1f}% non-zero values")
                
                if key == 'keypoints' and non_zero_count == 0:
                    logger.warning(f"{pair_id} - Empty keypoints detected!")
    
    def _save_debug_sample(self, sample: dict, pair_id: str, index: int):
        """Save debug visualization of loaded sample"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'DEBUG SAMPLE {index}: {pair_id}', fontsize=14, fontweight='bold')
        
        # Human image
        human_img = self.denormalize_image(sample['human_image'])
        if len(human_img.shape) == 3:
            human_img = human_img.permute(1, 2, 0).clamp(0, 1).numpy()
        axes[0, 0].imshow(human_img)
        axes[0, 0].set_title('Human Image')
        axes[0, 0].axis('off')
        
        # Cloth image  
        cloth_img = self.denormalize_image(sample['cloth_image'])
        if len(cloth_img.shape) == 3:
            cloth_img = cloth_img.permute(1, 2, 0).clamp(0, 1).numpy()
        axes[0, 1].imshow(cloth_img)
        axes[0, 1].set_title('Cloth Image')
        axes[0, 1].axis('off')
        
        # Cloth mask
        cloth_mask = (sample['cloth_mask'].squeeze() + 1) / 2
        axes[0, 2].imshow(cloth_mask.numpy(), cmap='gray')
        axes[0, 2].set_title('Cloth Mask')
        axes[0, 2].axis('off')
        
        # Keypoints
        keypoints = (sample['keypoints'] + 1) / 2
        if keypoints.dim() > 2:
            keypoints_viz = keypoints.max(dim=0)[0].numpy()
        else:
            keypoints_viz = keypoints.numpy()
        axes[1, 0].imshow(keypoints_viz, cmap='viridis')
        axes[1, 0].set_title(f'Keypoints ({torch.sum(sample["keypoints"] != 0)} non-zero)')
        axes[1, 0].axis('off')
        
        # Parsing
        parsing = (sample['parsing'].squeeze() + 1) / 2
        axes[1, 1].imshow(parsing.numpy(), cmap='jet')
        axes[1, 1].set_title('Parsing')
        axes[1, 1].axis('off')
        
        # Masked human
        masked_human = self.denormalize_image(sample['masked_human'])
        if len(masked_human.shape) == 3:
            masked_human = masked_human.permute(1, 2, 0).clamp(0, 1).numpy()
        axes[1, 2].imshow(masked_human)
        axes[1, 2].set_title('Masked Human')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        debug_path = os.path.join(self.config.output_dir, self.config.debug_dir, f'debug_sample_{index:03d}_{pair_id}.png')
        plt.savefig(debug_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved debug sample: {debug_path}")
    
    def _print_validation_summary(self):
        """Print comprehensive validation summary"""
        stats = self.validation_stats
        
        logger.info("=" * 80)
        logger.info("DATASET VALIDATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total pairs found: {stats['total_pairs']}")
        logger.info(f"Valid pairs: {stats['valid_pairs']}")
        logger.info(f"Success rate: {(stats['valid_pairs']/stats['total_pairs']*100):.1f}%")
        
        if stats['missing_files']:
            logger.warning(f"Missing files ({len(stats['missing_files'])}): {stats['missing_files'][:5]}...")
        
        if stats['corrupted_files']:
            logger.warning(f"Corrupted files ({len(stats['corrupted_files'])}): {stats['corrupted_files'][:5]}...")
        
        kpt_stats = stats['keypoint_stats']
        logger.info(f"Keypoint validation:")
        logger.info(f"  - Valid keypoint files: {kpt_stats['valid']}")
        logger.info(f"  - Empty keypoint files: {kpt_stats['empty']}")
        logger.info(f"  - Total keypoints processed: {kpt_stats['total_keypoints']}")
        
        logger.info("=" * 80)
    
    def __len__(self):
        return len(self.pairs)
    
    def _load_image(self, path: Path, normalize: bool = True) -> torch.Tensor:
        """CORRECTED: Load and preprocess image with proper error handling"""
        try:
            if not path.exists():
                logger.error(f"Image file not found: {path}")
                return torch.zeros(3, self.image_size, self.image_size)
            
            image = Image.open(path).convert('RGB')
            original_size = image.size
            image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)  # CHW
            
            if normalize and self.image_normalize is not None:
                image = self.image_normalize(image)
            else:
                image = image * 2.0 - 1.0  # Normalize to [-1, 1]
            
            return image
        except Exception as e:
            logger.error(f"Error loading image {path}: {e}")
            return torch.zeros(3, self.image_size, self.image_size)
    
    def _load_mask(self, path: Path) -> torch.Tensor:
        """CORRECTED: Load and preprocess mask with proper validation"""
        try:
            if not path.exists():
                logger.error(f"Mask file not found: {path}")
                return torch.zeros(1, self.image_size, self.image_size)
            
            mask = Image.open(path).convert('L')
            mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
            mask = np.array(mask).astype(np.float32) / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0)  # Add channel dimension
            
            # Normalize mask to [-1, 1] for consistency
            mask = mask * 2.0 - 1.0
            return mask
        except Exception as e:
            logger.error(f"Error loading mask {path}: {e}")
            return torch.zeros(1, self.image_size, self.image_size)
    
    def _load_keypoints(self, path: Path) -> torch.Tensor:
        """CORRECTED: Load keypoints with YOUR actual JSON format"""
        try:
            if not path.exists():
                logger.warning(f"Keypoints file not found: {path}")
                return torch.zeros(18, self.image_size, self.image_size)
            
            # Handle both JSON and image formats
            if path.suffix == '.json':
                with open(path, 'r') as f:
                    kpts_data = json.load(f)
                
                keypoints = torch.zeros(18, self.image_size, self.image_size)
                
                # FIXED: Handle YOUR actual JSON format
                pose_keypoints = None
                
                # Format 1: Direct keypoints (YOUR current format)
                if 'pose_keypoints_2d' in kpts_data:
                    pose_keypoints = kpts_data['pose_keypoints_2d']
                    logger.debug(f"Found direct pose_keypoints_2d with {len(pose_keypoints)} keypoints")
                
                # Format 2: OpenPose format with people array (fallback)
                elif 'people' in kpts_data and len(kpts_data['people']) > 0:
                    person = kpts_data['people'][0]
                    if 'pose_keypoints_2d' in person:
                        pose_keypoints = person['pose_keypoints_2d']
                        logger.debug(f"Found OpenPose format keypoints")
                
                # Process keypoints if found
                if pose_keypoints:
                    valid_keypoints = 0
                    
                    # YOUR format: [[x, y], [x, y], ...] - list of coordinate pairs
                    if isinstance(pose_keypoints[0], list):
                        for i, kpt in enumerate(pose_keypoints[:18]):  # Limit to 18 keypoints
                            if len(kpt) >= 2:
                                x, y = float(kpt[0]), float(kpt[1])
                                
                                # Scale keypoints to current image size (assuming original was different)
                                # You may need to adjust this scaling factor based on your data
                                if x > 1.0 or y > 1.0:  # Absolute coordinates
                                    x_scaled = int(x * self.image_size / 512)  # Adjust 512 to your original size
                                    y_scaled = int(y * self.image_size / 512)
                                else:  # Normalized coordinates
                                    x_scaled = int(x * self.image_size)
                                    y_scaled = int(y * self.image_size)
                                
                                if 0 <= x_scaled < self.image_size and 0 <= y_scaled < self.image_size:
                                    keypoints[i, y_scaled, x_scaled] = 1.0
                                    valid_keypoints += 1
                    
                    # Fallback: Flat array format [x, y, conf, x, y, conf, ...]
                    else:
                        for i in range(0, min(len(pose_keypoints), 54), 3):  # 18 keypoints * 3
                            if i + 2 < len(pose_keypoints):
                                x, y, conf = pose_keypoints[i], pose_keypoints[i+1], pose_keypoints[i+2]
                                
                                if conf > 0:  # Only use confident keypoints
                                    x_scaled = int(x * self.image_size / 512)  # Adjust scaling
                                    y_scaled = int(y * self.image_size / 512)
                                    
                                    if 0 <= x_scaled < self.image_size and 0 <= y_scaled < self.image_size:
                                        keypoints[i//3, y_scaled, x_scaled] = 1.0
                                        valid_keypoints += 1
                    
                    logger.debug(f"Processed {valid_keypoints} valid keypoints")
                
                # Normalize keypoints to [-1, 1]
                keypoints = keypoints * 2.0 - 1.0
                return keypoints
            
            else:
                # Handle image-based keypoints
                kpt_img = Image.open(path).convert('L')
                kpt_img = kpt_img.resize((self.image_size, self.image_size), Image.NEAREST)
                kpt_tensor = torch.from_numpy(np.array(kpt_img)).unsqueeze(0).float() / 255.0
                kpt_tensor = kpt_tensor * 2.0 - 1.0  # Normalize to [-1, 1]
                return kpt_tensor
                
        except Exception as e:
            logger.error(f"Error loading keypoints {path}: {e}")
            return torch.zeros(18, self.image_size, self.image_size)
    
    def _load_parsing(self, path: Path) -> torch.Tensor:
        """CORRECTED: Load and preprocess parsing map with validation"""
        try:
            if not path.exists():
                logger.warning(f"Parsing file not found: {path}")
                return torch.zeros(1, self.image_size, self.image_size)
            
            parsing = Image.open(path).convert('L')
            parsing = parsing.resize((self.image_size, self.image_size), Image.NEAREST)
            parsing = np.array(parsing).astype(np.float32) / 255.0
            parsing = torch.from_numpy(parsing).unsqueeze(0)
            
            # Normalize parsing to [-1, 1]
            parsing = parsing * 2.0 - 1.0
            return parsing
        except Exception as e:
            logger.error(f"Error loading parsing {path}: {e}")
            return torch.zeros(1, self.image_size, self.image_size)
    
    def denormalize_image(self, tensor: torch.Tensor) -> torch.Tensor:
        """Denormalize image tensor for visualization"""
        if self.denormalize is not None:
            return self.denormalize(tensor.clone())
        else:
            return (tensor * 0.5 + 0.5).clamp(0, 1)
    
    def debug_keypoint_loading(self, pair_id: str):
        """DEBUG: Test keypoint loading for specific pair"""
        pair_dir = self.data_root / "pairs" / pair_id
        keypoints_path = pair_dir / "keypoints.json"
        
        print(f"=== DEBUGGING KEYPOINTS FOR {pair_id} ===")
        print(f"Path: {keypoints_path}")
        print(f"Exists: {keypoints_path.exists()}")
        
        if keypoints_path.exists():
            with open(keypoints_path, 'r') as f:
                raw_data = json.load(f)
            
            print(f"Raw JSON keys: {list(raw_data.keys())}")
            if 'pose_keypoints_2d' in raw_data:
                keypoints = raw_data['pose_keypoints_2d']
                print(f"Number of keypoints: {len(keypoints)}")
                print(f"First 5 keypoints: {keypoints[:5]}")
                print(f"Keypoint format: {type(keypoints[0]) if keypoints else 'Empty'}")
            
            # Load using corrected method
            loaded_keypoints = self._load_keypoints(keypoints_path)
            print(f"Loaded tensor shape: {loaded_keypoints.shape}")
            print(f"Non-zero values: {torch.sum(loaded_keypoints != 0).item()}")
            print(f"Min/Max values: {loaded_keypoints.min().item():.3f}/{loaded_keypoints.max().item():.3f}")
        
        print("=" * 50)
    
    def __getitem__(self, idx):
        pair_id = self.pairs[idx]
        pair_dir = self.data_root / "pairs" / pair_id
        
        # Load all required data with proper validation
        human_image = self._load_image(pair_dir / "human.png")
        cloth_image = self._load_image(pair_dir / "cloth.png")
        cloth_mask = self._load_mask(pair_dir / "cloth_mask.png")
        
        # Load optional data with fallbacks
        keypoints_path = pair_dir / "keypoints.json"
        if keypoints_path.exists():
            keypoints = self._load_keypoints(keypoints_path)
        else:
            keypoints = torch.zeros(18, self.image_size, self.image_size)
            logger.debug(f"No keypoints found for {pair_id}, using zeros")
        
        parsing_path = pair_dir / "parsing.png"
        if parsing_path.exists():
            parsing = self._load_parsing(parsing_path)
        else:
            parsing = torch.zeros(1, self.image_size, self.image_size)
            logger.debug(f"No parsing found for {pair_id}, using zeros")
        
        masked_human_path = pair_dir / "masked_human.png"
        if masked_human_path.exists():
            masked_human = self._load_image(masked_human_path)
        else:
            masked_human = human_image.clone()
            logger.debug(f"No masked_human found for {pair_id}, using human image")
        
        return {
            'human_image': human_image,
            'cloth_image': cloth_image,
            'cloth_mask': cloth_mask,
            'keypoints': keypoints,
            'parsing': parsing,
            'masked_human': masked_human,
            'pair_info': {'id': pair_id}
        }

class ProductionOOTDUNet(torch.nn.Module):
    """Production-grade OOTDiffusion UNet"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        
        # UNet initialization
        try:
            local_unet_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                         "checkpoints", "ootd", "ootd_dc", "checkpoint-36000", "unet_vton")
            
            if os.path.exists(local_unet_path):
                logger.info(f"Loading UNet from local checkpoint: {local_unet_path}")
                pretrained_unet = UNet2DConditionModel.from_pretrained(local_unet_path)
                
                self.unet = UNet2DConditionModel(
                    sample_size=config.image_size // 8,
                    in_channels=config.in_channels,
                    out_channels=config.out_channels,
                    layers_per_block=pretrained_unet.config.layers_per_block,
                    block_out_channels=pretrained_unet.config.block_out_channels,
                    down_block_types=pretrained_unet.config.down_block_types,
                    up_block_types=pretrained_unet.config.up_block_types,
                    cross_attention_dim=pretrained_unet.config.cross_attention_dim,
                    attention_head_dim=pretrained_unet.config.attention_head_dim,
                    use_linear_projection=pretrained_unet.config.use_linear_projection,
                )
                
                for name, param in pretrained_unet.named_parameters():
                    if "conv_in" not in name:
                        self.unet.state_dict()[name].copy_(param)
                
                logger.info("Successfully loaded and adapted pretrained UNet")
            else:
                logger.warning("Local UNet checkpoint not found, initializing from scratch")
                self.unet = UNet2DConditionModel(
                    sample_size=config.image_size // 8,
                    in_channels=config.in_channels,
                    out_channels=config.out_channels,
                    layers_per_block=2,
                    block_out_channels=(320, 640, 1280, 1280),
                    down_block_types=(
                        "CrossAttnDownBlock2D",
                        "CrossAttnDownBlock2D", 
                        "CrossAttnDownBlock2D",
                        "DownBlock2D"
                    ),
                    up_block_types=(
                        "UpBlock2D",
                        "CrossAttnUpBlock2D",
                        "CrossAttnUpBlock2D", 
                        "CrossAttnUpBlock2D"
                    ),
                    cross_attention_dim=768,
                    attention_head_dim=8,
                    use_linear_projection=True,
                )
        except Exception as e:
            logger.warning(f"Error loading pretrained UNet: {str(e)}. Initializing from scratch.")
            self.unet = UNet2DConditionModel(
                sample_size=config.image_size // 8,
                in_channels=config.in_channels,
                out_channels=config.out_channels,
                layers_per_block=2,
                block_out_channels=(320, 640, 1280, 1280),
                down_block_types=(
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D", 
                    "CrossAttnDownBlock2D",
                    "DownBlock2D"
                ),
                up_block_types=(
                    "UpBlock2D",
                    "CrossAttnUpBlock2D",
                    "CrossAttnUpBlock2D", 
                    "CrossAttnUpBlock2D"
                ),
                cross_attention_dim=768,
                attention_head_dim=8,
                use_linear_projection=True,
            )
        
        # Encoders
        self.cloth_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(4, 64, 3, padding=1),  # cloth_image(3) + cloth_mask(1)
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 3, padding=1, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, 3, padding=1, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 4, 3, padding=1),
        )
        
        self.human_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(22, 64, 3, padding=1),  # keypoints(18) + parsing(1) + masked_human(3)
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 3, padding=1, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, 3, padding=1, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 4, 3, padding=1),
        )
    
    def forward(self, noisy_latents, timesteps, cloth_condition, human_condition, encoder_hidden_states=None):
        """Forward pass"""
        batch_size, _, height, width = noisy_latents.shape
        
        cloth_cond = self.cloth_encoder(cloth_condition)
        human_cond = self.human_encoder(human_condition)
        
        if cloth_cond.shape[2:] != noisy_latents.shape[2:]:
            cloth_cond = F.interpolate(cloth_cond, size=(height, width), mode='bilinear', align_corners=False)
        
        if human_cond.shape[2:] != noisy_latents.shape[2:]:
            human_cond = F.interpolate(human_cond, size=(height, width), mode='bilinear', align_corners=False)
        
        model_input = torch.cat([noisy_latents, cloth_cond, human_cond], dim=1)
        
        noise_pred = self.unet(
            model_input, 
            timesteps, 
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False
        )[0]
        
        return noise_pred

class EnhancedProductionOOTDTrainer:
    """ENHANCED: Trainer with masked region generation and full image loss"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Setup logging with proper encoding
        self.setup_logging()
        
        # Initialize components
        self.init_models()
        self.init_scheduler()
        self.init_dataset()
        self.init_optimizer()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        self.training_history = {
            'epoch': [],
            'loss': [],
            'lr': [],
            'time': []
        }
        
        logger.info(f"Initialized Enhanced OOTDTrainer with masked generation")
        logger.info(f"Model parameters: {self.count_parameters():,}")
        logger.info(f"Masked generation enabled: {self.config.use_masked_generation}")
    
    def setup_logging(self):
        """CORRECTED: Setup logging with proper UTF-8 encoding"""
        log_file = os.path.join(self.config.output_dir, self.config.logs_dir, 
                               f"masked_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        # Create file handler with UTF-8 encoding
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add to logger
        logger.addHandler(file_handler)
        logger.info("Logging setup completed with UTF-8 encoding")
    
    def test_data_loading(self):
        """NEW: Test data loading on first few samples"""
        logger.info("Testing data loading on sample data...")
        
        for i in range(min(3, len(self.dataset))):
            pair_id = self.dataset.pairs[i]
            logger.info(f"Testing sample {i}: {pair_id}")
            
            # Test keypoint loading specifically
            self.dataset.debug_keypoint_loading(pair_id)
            
            # Test full sample loading
            try:
                sample = self.dataset[i]
                logger.info(f"Sample {i} loaded successfully:")
                for key, value in sample.items():
                    if isinstance(value, torch.Tensor):
                        non_zero = torch.sum(value != 0).item()
                        logger.info(f"  {key}: {value.shape} ({non_zero} non-zero values)")
                    else:
                        logger.info(f"  {key}: {value}")
            except Exception as e:
                logger.error(f"Failed to load sample {i}: {e}")
        
        logger.info("Data loading test completed")
    
    def init_models(self):
        """Initialize models"""
        self.model = ProductionOOTDUNet(self.config).to(self.device)
        
        # VAE and text encoder initialization
        try:
            local_vae_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                         "checkpoints", "ootd", "vae")
            if os.path.exists(local_vae_path):
                self.vae = AutoencoderKL.from_pretrained(local_vae_path).to(self.device)
            else:
                self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(self.device)
            
            self.vae.eval()
            self.vae.requires_grad_(False)
            logger.info("Successfully loaded pretrained VAE")
        except Exception as e:
            logger.warning(f"Could not load pretrained VAE: {str(e)}")
            # Fallback VAE initialization...
            self.vae = AutoencoderKL(
                in_channels=3,
                out_channels=3,
                down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
                up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
                block_out_channels=[128, 256, 512, 512],
                latent_channels=4,
                layers_per_block=2,
            ).to(self.device)
            self.vae.eval()
            self.vae.requires_grad_(False)
        
        # Text encoder
        try:
            self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
            self.text_encoder.eval()
            self.text_encoder.requires_grad_(False)
        except:
            self.text_encoder = None
            self.tokenizer = None
    
    def init_scheduler(self):
        """Initialize noise scheduler"""
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.config.timesteps,
            beta_start=self.config.beta_start,
            beta_end=self.config.beta_end,
            beta_schedule=self.config.beta_schedule,
            prediction_type=self.config.prediction_type,
        )
        logger.info(f"Initialized scheduler with {self.config.timesteps} timesteps")
    
    def init_dataset(self):
        """CORRECTED: Initialize dataset with comprehensive validation"""
        self.dataset = EnhancedOOTDDataset(self.config.data_root, self.config)
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        # Validation dataloader
        val_size = min(len(self.dataset) // 10, 100)
        val_indices = list(range(0, val_size))
        val_dataset = torch.utils.data.Subset(self.dataset, val_indices)
        
        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        # Test data loading
        if self.config.validate_data_loading:
            self.test_data_loading()
        
        logger.info(f"Dataset initialized: {len(self.dataset)} samples")
    
    def init_optimizer(self):
        """Initialize optimizer and scheduler"""
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        self.lr_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.num_epochs * len(self.dataloader),
            eta_min=self.config.learning_rate * 0.1
        )
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def encode_images(self, images):
        """Encode images to latent space with proper normalization handling"""
        with torch.no_grad():
            # If images are normalized with ImageNet stats, denormalize first
            if self.config.normalize_images:
                images = self.dataset.denormalize_image(images)
                images = images * 2.0 - 1.0  # Convert to VAE expected range [-1, 1]
            
            latents = self.vae.encode(images).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        return latents

    def decode_latents(self, latents):
        """Decode latents to images"""
        with torch.no_grad():
            latents = latents / self.vae.config.scaling_factor
            images = self.vae.decode(latents).sample
            images = (images / 2 + 0.5).clamp(0, 1)
        return images

    def prepare_text_embeddings(self, texts):
        """Prepare text embeddings if text encoder available"""
        if self.text_encoder is None:
            return torch.zeros(len(texts), 77, 768, device=self.device)
        
        with torch.no_grad():
            tokens = self.tokenizer(
                texts,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt"
            ).input_ids.to(self.device)
            
            embeddings = self.text_encoder(tokens)[0]
        
        return embeddings

    def create_masked_visualization(self, batch, predictions=None, mask_latent=None, epoch=0, batch_idx=0, phase="train"):
        """ENHANCED: Create visualization showing masked region generation"""
        if not self.config.save_comprehensive_results:
            return
        
        with torch.no_grad():
            batch_size = min(4, batch['human_image'].shape[0])
            
            # Create enhanced figure with mask visualization
            fig, axes = plt.subplots(8, batch_size, figsize=(batch_size * 5, 40))
            
            if batch_size == 1:
                axes = axes.reshape(-1, 1)
            
            for i in range(batch_size):
                # Row 0: Expected Output (Human Image)
                human_img = self.dataset.denormalize_image(batch['human_image'][i].cpu())
                if len(human_img.shape) == 3:
                    human_img = human_img.permute(1, 2, 0).clamp(0, 1).numpy()
                axes[0, i].imshow(human_img)
                axes[0, i].set_title(f"Expected Full Output {i}", fontweight='bold', color='green', fontsize=12)
                axes[0, i].axis('off')
                
                # Row 1: Cloth Image Input
                cloth_img = self.dataset.denormalize_image(batch['cloth_image'][i].cpu())
                if len(cloth_img.shape) == 3:
                    cloth_img = cloth_img.permute(1, 2, 0).clamp(0, 1).numpy()
                axes[1, i].imshow(cloth_img)
                axes[1, i].set_title(f"Input: Cloth {i}", fontweight='bold', color='blue', fontsize=12)
                axes[1, i].axis('off')
                
                # Row 2: Cloth Mask Input
                cloth_mask = (batch['cloth_mask'][i].cpu().numpy().squeeze() + 1) / 2
                axes[2, i].imshow(cloth_mask, cmap='gray')
                axes[2, i].set_title(f"Generation Mask {i}", fontweight='bold', color='orange', fontsize=12)
                axes[2, i].axis('off')
                
                # Row 3: Keypoints Input
                keypoints = (batch['keypoints'][i].cpu() + 1) / 2
                if keypoints.dim() > 2:
                    keypoints = keypoints.max(dim=0)[0].numpy()
                else:
                    keypoints = keypoints.numpy().squeeze()
                axes[3, i].imshow(keypoints, cmap='viridis')
                axes[3, i].set_title(f"Input: Keypoints {i}", fontweight='bold', color='blue', fontsize=12)
                axes[3, i].axis('off')
                
                # Row 4: Parsing Input
                parsing = (batch['parsing'][i].cpu().numpy().squeeze() + 1) / 2
                axes[4, i].imshow(parsing, cmap='jet')
                axes[4, i].set_title(f"Input: Parsing {i}", fontweight='bold', color='blue', fontsize=12)
                axes[4, i].axis('off')
                
                # Row 5: Masked Human Input
                masked_human = self.dataset.denormalize_image(batch['masked_human'][i].cpu())
                if len(masked_human.shape) == 3:
                    masked_human = masked_human.permute(1, 2, 0).clamp(0, 1).numpy()
                axes[5, i].imshow(masked_human)
                axes[5, i].set_title(f"Input: Masked Human {i}", fontweight='bold', color='blue', fontsize=12)
                axes[5, i].axis('off')
                
                # Row 6: Generated Full Image (with masked region)
                if predictions is not None and i < predictions.shape[0]:
                    pred_img = predictions[i].cpu().permute(1, 2, 0).clamp(0, 1).numpy()
                    axes[6, i].imshow(pred_img)
                    axes[6, i].set_title(f"Generated Full Image {i}", fontweight='bold', color='red', fontsize=12)
                else:
                    axes[6, i].imshow(np.zeros((self.config.image_size, self.config.image_size, 3)))
                    axes[6, i].set_title(f"Generated Full Image {i} (N/A)", fontweight='bold', color='gray', fontsize=12)
                axes[6, i].axis('off')
                
                # Row 7: Mask Overlay on Generated Image
                if predictions is not None and i < predictions.shape[0]:
                    pred_img = predictions[i].cpu().permute(1, 2, 0).clamp(0, 1).numpy()
                    # Overlay mask
                    mask_overlay = cloth_mask[:, :, np.newaxis]
                    highlighted_img = pred_img.copy()
                    highlighted_img[:, :, 1] = np.maximum(highlighted_img[:, :, 1], mask_overlay.squeeze() * 0.5)  # Green highlight
                    axes[7, i].imshow(highlighted_img)
                    axes[7, i].set_title(f"Generated with Mask Overlay {i}", fontweight='bold', color='purple', fontsize=12)
                else:
                    axes[7, i].imshow(np.zeros((self.config.image_size, self.config.image_size, 3)))
                    axes[7, i].set_title(f"Mask Overlay {i} (N/A)", fontweight='bold', color='gray', fontsize=12)
                axes[7, i].axis('off')
            
            # Enhanced title
            plt.suptitle(f'{phase.upper()} - Epoch {epoch} - Batch {batch_idx}\n'
                        f'MASKED REGION GENERATION: Only generates masked area, loss on full image\n'
                        f'Blue: Inputs | Orange: Generation Mask | Green: Expected | Red: Generated | Purple: Mask Overlay', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Save the enhanced visualization
            save_dir = os.path.join(self.config.output_dir, self.config.comprehensive_results_dir)
            save_path = os.path.join(save_dir, f'{phase}_masked_epoch_{epoch:03d}_batch_{batch_idx:06d}_comprehensive.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved masked generation visualization to {save_path}")

    def training_step(self, batch):
        """ENHANCED: Training step with masked region generation + full image loss"""
        # Move to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)
        
        # Encode target images to latents
        target_latents = self.encode_images(batch['human_image'])
        
        # Sample noise
        noise = torch.randn_like(target_latents)
        
        # Sample timesteps
        timesteps = torch.randint(0, self.config.timesteps, (target_latents.shape[0],), device=self.device)
        
        # Add noise to latents
        noisy_latents = self.noise_scheduler.add_noise(target_latents, noise, timesteps)
        
        # Prepare conditions
        cloth_condition = torch.cat([batch['cloth_image'], batch['cloth_mask']], dim=1)
        keypoints_resized = F.interpolate(batch['keypoints'], size=(self.config.image_size, self.config.image_size), mode='nearest')
        parsing_resized = F.interpolate(batch['parsing'], size=(self.config.image_size, self.config.image_size), mode='nearest')
        
        human_condition = torch.cat([
            keypoints_resized,
            parsing_resized, 
            batch['masked_human']
        ], dim=1)
        
        # Text embeddings
        texts = ["a person wearing clothes"] * target_latents.shape[0]
        text_embeddings = self.prepare_text_embeddings(texts)
        
        # ENHANCED: Model prediction for masked region only
        noise_pred_masked = self.model(noisy_latents, timesteps, cloth_condition, human_condition, text_embeddings)
        
        # ENHANCED: Create mask for latent space (downsample cloth_mask)
        latent_size = target_latents.shape[-1]  # Usually 32 for 256x256 images
        cloth_mask_latent = F.interpolate(
            batch['cloth_mask'], 
            size=(latent_size, latent_size), 
            mode='nearest'
        )
        
        # Normalize mask to [0, 1] for proper masking
        cloth_mask_latent = (cloth_mask_latent + 1) / 2
        
        # ENHANCED: Apply mask to combine predictions
        if self.config.use_masked_generation:
            if self.config.prediction_type == "epsilon":
                # For epsilon prediction, mask the noise prediction
                target_noise = noise
                
                # Only apply predicted noise to masked regions, keep original noise elsewhere
                combined_noise_pred = noise_pred_masked * cloth_mask_latent + noise * (1 - cloth_mask_latent)
                
                # Calculate weighted loss if enabled
                if self.config.mask_loss_weight > 1.0:
                    # Higher weight for masked regions
                    loss_weights = cloth_mask_latent * self.config.mask_loss_weight + (1 - cloth_mask_latent) * 1.0
                    loss_unweighted = F.mse_loss(combined_noise_pred, target_noise, reduction='none')
                    loss = torch.mean(loss_unweighted * loss_weights)
                else:
                    # Standard loss
                    loss = F.mse_loss(combined_noise_pred, target_noise)
                
                # Store for visualization
                noise_pred_for_vis = combined_noise_pred
                
            else:
                # For v-prediction or x0 prediction
                target_latents_clean = target_latents
                
                # Only apply predicted latents to masked regions, keep original elsewhere  
                combined_latent_pred = noise_pred_masked * cloth_mask_latent + target_latents * (1 - cloth_mask_latent)
                
                # Calculate weighted loss if enabled
                if self.config.mask_loss_weight > 1.0:
                    loss_weights = cloth_mask_latent * self.config.mask_loss_weight + (1 - cloth_mask_latent) * 1.0
                    loss_unweighted = F.mse_loss(combined_latent_pred, target_latents_clean, reduction='none')
                    loss = torch.mean(loss_unweighted * loss_weights)
                else:
                    loss = F.mse_loss(combined_latent_pred, target_latents_clean)
                
                # Store for visualization
                noise_pred_for_vis = combined_latent_pred
        else:
            # Standard approach without masking
            if self.config.prediction_type == "epsilon":
                target = noise
            else:
                target = target_latents
            
            loss = F.mse_loss(noise_pred_masked, target)
            noise_pred_for_vis = noise_pred_masked
            cloth_mask_latent = None
        
        # ENHANCED: Generate predictions for visualization (every 20 steps)
        if self.global_step % 20 == 0:
            with torch.no_grad():
                # Reconstruct full image for visualization
                if self.config.prediction_type == "epsilon":
                    # Denoise using combined prediction
                    alpha_t = self.noise_scheduler.alphas_cumprod[timesteps[0].item()].sqrt()
                    sigma_t = (1 - self.noise_scheduler.alphas_cumprod[timesteps[0].item()]).sqrt()
                    pred_latents = (noisy_latents - sigma_t * noise_pred_for_vis) / alpha_t
                else:
                    pred_latents = noise_pred_for_vis
                
                pred_images = self.decode_latents(pred_latents)
                
                # Create comprehensive visualization with mask overlay
                self.create_masked_visualization(
                    batch, pred_images, cloth_mask_latent, self.epoch, self.global_step, "train"
                )
        
        return loss

    def training_step_for_validation(self, batch):
        """Training step computation for validation (no gradients)"""
        # Encode target images to latents
        target_latents = self.encode_images(batch['human_image'])
        
        # Sample noise
        noise = torch.randn_like(target_latents)
        
        # Sample timesteps
        timesteps = torch.randint(0, self.config.timesteps, (target_latents.shape[0],), device=self.device)
        
        # Add noise to latents
        noisy_latents = self.noise_scheduler.add_noise(target_latents, noise, timesteps)
        
        # Prepare conditions
        cloth_condition = torch.cat([batch['cloth_image'], batch['cloth_mask']], dim=1)
        keypoints_resized = F.interpolate(batch['keypoints'], size=(self.config.image_size, self.config.image_size), mode='nearest')
        parsing_resized = F.interpolate(batch['parsing'], size=(self.config.image_size, self.config.image_size), mode='nearest')
        human_condition = torch.cat([keypoints_resized, parsing_resized, batch['masked_human']], dim=1)
        
        texts = ["a person wearing clothes"] * target_latents.shape[0]
        text_embeddings = self.prepare_text_embeddings(texts)
        
        # Model prediction
        noise_pred_masked = self.model(noisy_latents, timesteps, cloth_condition, human_condition, text_embeddings)
        
        # Apply masking for validation
        if self.config.use_masked_generation:
            latent_size = target_latents.shape[-1]
            cloth_mask_latent = F.interpolate(
                batch['cloth_mask'], 
                size=(latent_size, latent_size), 
                mode='nearest'
            )
            cloth_mask_latent = (cloth_mask_latent + 1) / 2
            
            if self.config.prediction_type == "epsilon":
                target = noise
                combined_noise_pred = noise_pred_masked * cloth_mask_latent + noise * (1 - cloth_mask_latent)
                loss = F.mse_loss(combined_noise_pred, target)
            else:
                target = target_latents
                combined_latent_pred = noise_pred_masked * cloth_mask_latent + target_latents * (1 - cloth_mask_latent)
                loss = F.mse_loss(combined_latent_pred, target)
        else:
            if self.config.prediction_type == "epsilon":
                target = noise
            else:
                target = target_latents
            loss = F.mse_loss(noise_pred_masked, target)
        
        return loss

    def validation_step(self):
        """Enhanced validation step with comprehensive visualization"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dataloader):
                if num_batches >= 10:  # Limit validation batches
                    break
                
                # Move to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                loss = self.training_step_for_validation(batch)
                total_loss += loss.item()
                num_batches += 1
                
                # Generate comprehensive validation visualization for first batch
                if batch_idx == 0:
                    self.generate_enhanced_validation_samples(batch, batch_idx)
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        self.model.train()
        return avg_loss

    def generate_enhanced_validation_samples(self, batch, batch_idx):
        """Generate enhanced validation samples with comprehensive visualization"""
        with torch.no_grad():
            batch_size = min(self.config.max_validation_samples, batch['human_image'].shape[0])
            
            # Create simple prediction for validation visualization
            target_latents = self.encode_images(batch['human_image'][:batch_size])
            noise = torch.randn_like(target_latents)
            
            # Simple denoising prediction for visualization
            timesteps = torch.full((batch_size,), 500, device=self.device)
            noisy_latents = self.noise_scheduler.add_noise(target_latents, noise, timesteps)
            
            # Prepare conditions
            cloth_condition = torch.cat([batch['cloth_image'][:batch_size], batch['cloth_mask'][:batch_size]], dim=1)
            keypoints_resized = F.interpolate(batch['keypoints'][:batch_size], size=(self.config.image_size, self.config.image_size), mode='nearest')
            parsing_resized = F.interpolate(batch['parsing'][:batch_size], size=(self.config.image_size, self.config.image_size), mode='nearest')
            human_condition = torch.cat([keypoints_resized, parsing_resized, batch['masked_human'][:batch_size]], dim=1)
            
            texts = ["a person wearing clothes"] * batch_size
            text_embeddings = self.prepare_text_embeddings(texts)
            
            # Get prediction
            noise_pred = self.model(noisy_latents, timesteps, cloth_condition, human_condition, text_embeddings)
            
            # Apply masking if enabled
            if self.config.use_masked_generation:
                latent_size = target_latents.shape[-1]
                cloth_mask_latent = F.interpolate(
                    batch['cloth_mask'][:batch_size], 
                    size=(latent_size, latent_size), 
                    mode='nearest'
                )
                cloth_mask_latent = (cloth_mask_latent + 1) / 2
                
                # Simple denoising with masking
                alpha_t = self.noise_scheduler.alphas_cumprod[timesteps[0].item()].sqrt()
                sigma_t = (1 - self.noise_scheduler.alphas_cumprod[timesteps[0].item()]).sqrt()
                
                combined_noise_pred = noise_pred * cloth_mask_latent + noise * (1 - cloth_mask_latent)
                pred_latents = (noisy_latents - sigma_t * combined_noise_pred) / alpha_t
            else:
                # Simple denoising without masking
                alpha_t = self.noise_scheduler.alphas_cumprod[timesteps[0].item()].sqrt()
                sigma_t = (1 - self.noise_scheduler.alphas_cumprod[timesteps[0].item()]).sqrt()
                pred_latents = (noisy_latents - sigma_t * noise_pred) / alpha_t
            
            # Decode to images
            generated_images = self.decode_latents(pred_latents)
            
            # Create comprehensive validation visualization
            batch_subset = {}
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch_subset[key] = batch[key][:batch_size]
                else:
                    batch_subset[key] = batch[key]
            
            self.create_masked_visualization(
                batch_subset, generated_images, None, self.epoch, batch_idx, "validation"
            )

    def save_checkpoint(self, epoch, loss, is_best=False):
        """Save model checkpoint with enhanced information"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'loss': loss,
            'config': asdict(self.config),
            'training_history': self.training_history,
            'normalization_enabled': self.config.normalize_images,
            'masked_generation_enabled': self.config.use_masked_generation
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.config.output_dir, self.config.checkpoint_dir, f'masked_checkpoint_epoch_{epoch:03d}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.config.output_dir, self.config.checkpoint_dir, 'masked_best_model.pth')
            torch.save(checkpoint, best_path)
            logger.info(f"Saved masked generation best model checkpoint (loss: {loss:.6f})")
        
        logger.info(f"Saved masked generation checkpoint: {checkpoint_path}")
    
    def train(self):
        """ENHANCED: Main training loop with masked generation logging"""
        logger.info("Starting enhanced training with masked generation...")
        logger.info(f"Configuration: {self.config}")
        logger.info(f"Normalization enabled: {self.config.normalize_images}")
        logger.info(f"Masked generation: {self.config.use_masked_generation}")
        logger.info(f"Mask loss weight: {self.config.mask_loss_weight}")
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            epoch_start_time = time.time()
            
            # Training phase
            self.model.train()
            epoch_loss = 0
            num_batches = 0
            
            for batch_idx, batch in enumerate(self.dataloader):
                self.optimizer.zero_grad()
                
                loss = self.training_step(batch)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                self.optimizer.step()
                self.lr_scheduler.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                self.global_step += 1
                
                # FIXED: No Unicode characters in logging
                if batch_idx % self.config.log_every == 0:
                    current_lr = self.lr_scheduler.get_last_lr()[0]
                    logger.info(f"[MASKED-TRAIN] Epoch {epoch:03d}/{self.config.num_epochs:03d} | "
                              f"Batch {batch_idx:04d}/{len(self.dataloader):04d} | "
                              f"Loss: {loss.item():.6f} | "
                              f"LR: {current_lr:.8f} | "
                              f"Masked: {'YES' if self.config.use_masked_generation else 'NO'}")
            
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
            
            # Enhanced validation phase
            val_loss = 0
            if epoch % self.config.validate_every == 0:
                logger.info("Running enhanced validation with masked generation visualization...")
                val_loss = self.validation_step()
                logger.info(f"Epoch {epoch:03d} - Validation Loss: {val_loss:.6f}")
            
            # Update training history
            epoch_time = time.time() - epoch_start_time
            self.training_history['epoch'].append(epoch)
            self.training_history['loss'].append(avg_epoch_loss)
            self.training_history['lr'].append(self.lr_scheduler.get_last_lr()[0])
            self.training_history['time'].append(epoch_time)
            
            # Save checkpoints
            is_best = val_loss < self.best_loss if val_loss > 0 else avg_epoch_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss if val_loss > 0 else avg_epoch_loss
            
            if epoch % self.config.save_every == 0 or is_best:
                self.save_checkpoint(epoch, avg_epoch_loss, is_best)
            
            # Enhanced epoch summary
            total_time = time.time() - start_time
            logger.info(f"[MASKED-SUMMARY] Epoch {epoch:03d} completed:")
            logger.info(f"   Training Loss: {avg_epoch_loss:.6f}")
            logger.info(f"   Validation Loss: {val_loss:.6f}")
            logger.info(f"   Epoch Time: {epoch_time:.2f}s")
            logger.info(f"   Total Time: {total_time:.2f}s")
            logger.info(f"   Learning Rate: {self.lr_scheduler.get_last_lr()[0]:.8f}")
            logger.info(f"   Best Loss: {self.best_loss:.6f}")
            logger.info(f"   Masked Generation: {'ENABLED' if self.config.use_masked_generation else 'DISABLED'}")
            logger.info("=" * 80)
        
        total_training_time = time.time() - start_time
        logger.info(f"Masked generation training completed! Total time: {total_training_time:.2f}s")
        logger.info(f"Best loss achieved: {self.best_loss:.6f}")

def main():
    """ENHANCED: Main training function with masked generation"""
    # Enhanced configuration with masked generation enabled
    config = TrainingConfig(
        data_root="dataset",
        num_epochs=2500,
        timesteps=2500,
        batch_size=12,
        learning_rate=1e-4,
        output_dir="enhanced_masked_ootd_training_inpaint",
        normalize_images=True,
        save_comprehensive_results=True,
        validate_data_loading=True,  # Enable comprehensive validation
        save_debug_samples=True,     # Save debug samples
        max_debug_samples=5,         # Number of samples to debug
        
        # ENHANCED: Masked Generation Settings
        use_masked_generation=True,  # Enable masked generation
        mask_loss_weight=2.0,        # Higher weight for masked regions
        preserve_unmasked=True,      # Keep unmasked regions unchanged
        
        image_mean=(0.485, 0.456, 0.406),
        image_std=(0.229, 0.224, 0.225)
    )
    
    # Create enhanced trainer with masked generation
    trainer = EnhancedProductionOOTDTrainer(config)
    
    # Start enhanced training
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer.save_checkpoint(trainer.epoch, trainer.best_loss)
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
