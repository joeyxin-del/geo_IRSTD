"""
Optimized Training Script for Geo-IRSTD
Improved version of train_spotgeo.py with better code structure and readability
"""

import argparse
import time
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Local imports
from net import Net
from dataset_spotgeo import TrainSetLoader, TestSetLoader
from metrics import mIoU, PD_FA, SigmoidMetric, SamplewiseSigmoidMetric, F1Metric, MSEMetric
from CalculateFPS import FPSBenchmark

# SwanLab monitoring
try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False
    print("Warning: SwanLab not available. Install with: pip install swanlab")

# Set environment variables
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TrainingConfig:
    """Configuration class for training parameters"""
    
    def __init__(self):
        self.model_names: List[str] = ['WTNet']
        self.dataset_names: List[str] = ['spotgeov2-IRSTD']
        self.dataset_dir: str = './datasets'
        self.img_norm_cfg: Optional[Dict] = None
        
        # Training parameters
        self.batch_size: int = 2
        self.patch_size: int = 512
        self.num_epochs: int = 400
        self.seed: int = 42
        
        # Optimization
        self.optimizer_name: str = 'Adamw'
        self.learning_rate: float = 5e-4
        self.scheduler_name: str = 'CosineAnnealingLR'
        self.scheduler_settings: Dict = {'epochs': 800, 'min_lr': 0.0005}
        
        # System
        self.num_threads: int = 0
        self.save_dir: str = './log'
        self.resume_paths: Optional[List[str]] = None
        self.threshold: float = 0.5
        self.use_augmentation: bool = True
        self.use_orthogonal_reg: bool = False
        self.calculate_fps: bool = False
        
        # Advanced features
        self.use_amp: bool = False
        self.ema_decay: float = 0.0
        
        # Monitoring and checkpointing
        self.save_interval: int = 10  # Save checkpoint every N epochs
        self.val_interval: int = 5    # Validate every N epochs
        self.use_swanlab: bool = True  # Enable SwanLab monitoring
        
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'TrainingConfig':
        """Create config from command line arguments"""
        config = cls()
        config.model_names = args.model_names
        config.dataset_names = args.dataset_names
        config.dataset_dir = args.dataset_dir
        config.img_norm_cfg = args.img_norm_cfg
        config.batch_size = args.batch_size
        config.patch_size = args.patch_size
        config.num_epochs = args.num_epochs
        config.seed = args.seed
        config.optimizer_name = args.optimizer_name
        config.learning_rate = args.learning_rate
        config.scheduler_name = args.scheduler_name
        config.scheduler_settings = args.scheduler_settings
        config.num_threads = args.num_threads
        config.save_dir = args.save_dir
        config.resume_paths = args.resume_paths
        config.threshold = args.threshold
        config.use_augmentation = args.use_augmentation
        config.use_orthogonal_reg = args.use_orthogonal_reg
        config.calculate_fps = args.calculate_fps
        config.use_amp = args.use_amp
        config.ema_decay = args.ema_decay
        config.save_interval = args.save_interval
        config.val_interval = args.val_interval
        config.use_swanlab = args.use_swanlab
        return config


class ExponentialMovingAverage:
    """Exponential Moving Average for model parameters"""
    
    def __init__(self, model: nn.Module, decay: float):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
    def register(self):
        """Register model parameters for EMA"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                
    def update(self):
        """Update EMA parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
                
    def apply_shadow(self):
        """Apply EMA parameters to model"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
                
    def restore(self):
        """Restore original model parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class OptimizerFactory:
    """Factory class for creating optimizers and schedulers"""
    
    @staticmethod
    def get_optimizer(model: nn.Module, config: TrainingConfig) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """Create optimizer and scheduler based on configuration"""
        
        # Configure optimizer settings based on type
        optimizer_settings = OptimizerFactory._get_optimizer_settings(config)
        scheduler_settings = OptimizerFactory._get_scheduler_settings(config)
        
        # Create optimizer
        if config.optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), **optimizer_settings)
        elif config.optimizer_name == 'Adamw':
            optimizer = torch.optim.AdamW(model.parameters(), **optimizer_settings)
        elif config.optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), **optimizer_settings)
        elif config.optimizer_name == 'Adagrad':
            optimizer = torch.optim.Adagrad(model.parameters(), **optimizer_settings)
        else:
            raise ValueError(f"Unsupported optimizer: {config.optimizer_name}")
        
        # Create scheduler based on scheduler name and settings
        if config.scheduler_name == 'MultiStepLR':
            if 'step' not in scheduler_settings or 'gamma' not in scheduler_settings:
                raise ValueError(f"MultiStepLR requires 'step' and 'gamma' in scheduler settings")
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, 
                milestones=scheduler_settings['step'], 
                gamma=scheduler_settings['gamma']
            )
        elif config.scheduler_name == 'CosineAnnealingLR':
            if 'epochs' not in scheduler_settings:
                raise ValueError(f"CosineAnnealingLR requires 'epochs' in scheduler settings")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_settings['epochs'],
                eta_min=scheduler_settings.get('min_lr', 0)
            )
        else:
            raise ValueError(f"Unsupported scheduler: {config.scheduler_name}")
            
        return optimizer, scheduler
    
    @staticmethod
    def _get_optimizer_settings(config: TrainingConfig) -> Dict[str, Any]:
        """Get optimizer-specific settings"""
        base_settings = {'lr': config.learning_rate}
        
        if config.optimizer_name == 'Adam':
            return {'lr': 5e-4 * 4}
        elif config.optimizer_name == 'Adamw':
            return {'lr': 0.0015}
        elif config.optimizer_name == 'Lion':
            return {'lr': 0.00015}
        elif config.optimizer_name == 'Adagrad':
            return {'lr': 0.05}
        else:
            return base_settings
    
    @staticmethod
    def _get_scheduler_settings(config: TrainingConfig) -> Dict[str, Any]:
        """Get scheduler-specific settings"""
        # Get base settings based on optimizer
        if config.optimizer_name == 'Adam':
            base_settings = {'epochs': 400, 'step': [200, 300], 'gamma': 0.1}
        elif config.optimizer_name == 'Adamw':
            base_settings = {'epochs': 800, 'min_lr': 0.0005}
        elif config.optimizer_name == 'Lion':
            base_settings = {'epochs': 400, 'min_lr': 0.00001}
        elif config.optimizer_name == 'Adagrad':
            base_settings = {'epochs': 400, 'min_lr': 1e-3}
        else:
            base_settings = config.scheduler_settings
        
        # Ensure scheduler settings match the scheduler type
        if config.scheduler_name == 'MultiStepLR':
            # MultiStepLR needs 'step' and 'gamma'
            if 'step' not in base_settings:
                base_settings['step'] = [200, 300]
            if 'gamma' not in base_settings:
                base_settings['gamma'] = 0.1
        elif config.scheduler_name == 'CosineAnnealingLR':
            # CosineAnnealingLR needs 'epochs' and optionally 'min_lr'
            if 'epochs' not in base_settings:
                base_settings['epochs'] = 400
            if 'min_lr' not in base_settings:
                base_settings['min_lr'] = 0
        
        return base_settings


class ModelTrainer:
    """Main training class with improved structure"""
    
    def __init__(self, config: TrainingConfig, model_name: str, dataset_name: str):
        self.config = config
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training state
        self.best_f1 = -1.0
        self.current_epoch = 0
        self.total_loss_history = [0]
        
        # Setup logging
        self.setup_logging()
        
        # Setup SwanLab monitoring
        self.setup_swanlab()
        
        # Initialize components
        self.setup_data_loaders()
        self.setup_model()
        self.setup_training_components()
        
    def setup_logging(self):
        """Setup logging for this training run"""
        # 创建新的目录结构: ./log/{dataset_name}/{model_name}/timestamp/
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = Path(self.config.save_dir) / self.dataset_name / self.model_name / timestamp
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.model_dir = self.experiment_dir / "model"
        self.checkpoints_dir = self.experiment_dir / "checkpoints"
        self.model_dir.mkdir(exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)
        
        # 复制模型定义文件
        self._copy_model_files()
        
        # 保存实验配置
        self._save_config()
        
        # 设置日志文件
        log_file = self.experiment_dir / f"training_{timestamp}.log"
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
        self.log_file = log_file
        
        logger.info(f"Experiment directory created: {self.experiment_dir}")
        
    def _copy_model_files(self):
        """复制模型定义文件到model目录"""
        try:
            # 复制net.py文件
            net_source = Path("net.py")
            if net_source.exists():
                net_dest = self.model_dir / "net.py"
                import shutil
                shutil.copy2(net_source, net_dest)
                logger.info(f"Copied net.py to {net_dest}")
            
            # 复制dataset_spotgeo.py文件
            dataset_source = Path("dataset_spotgeo.py")
            if dataset_source.exists():
                dataset_dest = self.model_dir / "dataset_spotgeo.py"
                import shutil
                shutil.copy2(dataset_source, dataset_dest)
                logger.info(f"Copied dataset_spotgeo.py to {dataset_dest}")
            
            # 复制metrics.py文件
            metrics_source = Path("metrics.py")
            if metrics_source.exists():
                metrics_dest = self.model_dir / "metrics.py"
                import shutil
                shutil.copy2(metrics_source, metrics_dest)
                logger.info(f"Copied metrics.py to {metrics_dest}")
                
        except Exception as e:
            logger.warning(f"Failed to copy model files: {e}")
    
    def _save_config(self):
        """保存实验配置到config.json"""
        try:
            config_data = {
                'model_name': self.model_name,
                'dataset_name': self.dataset_name,
                'timestamp': time.strftime("%Y%m%d_%H%M%S"),
                'training_config': self.config.__dict__,
                'system_info': {
                    'device': str(self.device),
                    'cuda_available': torch.cuda.is_available(),
                    'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
                    'pytorch_version': torch.__version__
                }
            }
            
            config_file = self.experiment_dir / "config.json"
            import json
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Config saved to {config_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save config: {e}")
        
    def setup_swanlab(self):
        """Setup SwanLab monitoring"""
        if not self.config.use_swanlab or not SWANLAB_AVAILABLE:
            self.swanlab_run = None
            return
            
        try:
            # Create SwanLab run
            self.swanlab_run = swanlab.init(
                experiment_name=f"{self.model_name}_{self.dataset_name}",
                config={
                    "model_name": self.model_name,
                    "dataset_name": self.dataset_name,
                    "batch_size": self.config.batch_size,
                    "learning_rate": self.config.learning_rate,
                    "optimizer": self.config.optimizer_name,
                    "scheduler": self.config.scheduler_name,
                    "num_epochs": self.config.num_epochs,
                    "save_interval": self.config.save_interval,
                    "val_interval": self.config.val_interval,
                }
            )
            logger.info("SwanLab monitoring initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize SwanLab: {e}")
            self.swanlab_run = None
        
    def setup_data_loaders(self):
        """Setup training and validation data loaders"""
        # Training data
        train_dataset = TrainSetLoader(
            dataset_dir=self.config.dataset_dir,
            dataset_name=self.dataset_name,
            patch_size=self.config.patch_size,
            img_norm_cfg=self.config.img_norm_cfg
        )
        
        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_threads
        )
        
        # Validation data
        val_dataset = TestSetLoader(
            self.config.dataset_dir,
            self.dataset_name,
            self.dataset_name,
            patch_size=self.config.patch_size,
            img_norm_cfg=self.config.img_norm_cfg
        )
        
        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )
        
    def setup_model(self):
        """Setup the neural network model"""
        self.model = Net(model_name=self.model_name, mode='train').to(self.device)
        self.model.train()
        
    def setup_training_components(self):
        """Setup optimizer, scheduler, and other training components"""
        # Create optimizer and scheduler
        self.optimizer, self.scheduler = OptimizerFactory.get_optimizer(self.model, self.config)
        
        # Setup EMA
        self.ema = ExponentialMovingAverage(self.model, self.config.ema_decay)
        self.ema.register()
        
        # Setup AMP scaler if needed
        self.scaler = torch.cuda.amp.GradScaler() if self.config.use_amp else None
        
        # Load checkpoint if resuming
        self.load_checkpoint()
        
    def load_checkpoint(self):
        """Load checkpoint if resuming training"""
        if not self.config.resume_paths:
            return
            
        for resume_path in self.config.resume_paths:
            if self.model_name in resume_path and os.path.exists(resume_path):
                logger.info(f"Loading checkpoint from {resume_path}")
                try:
                    checkpoint = torch.load(resume_path, map_location=self.device)
                    self.model.load_state_dict(checkpoint['state_dict'])
                    self.current_epoch = checkpoint.get('epoch', 0)
                    self.total_loss_history = checkpoint.get('total_loss', [0])
                    self.best_f1 = checkpoint.get('best_f1', -1.0)
                    
                    # Load optimizer and scheduler state if available
                    if 'optimizer_state_dict' in checkpoint:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    if 'scheduler_state_dict' in checkpoint:
                        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                        
                    logger.info(f"Successfully loaded checkpoint from epoch {self.current_epoch}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load checkpoint from {resume_path}: {e}")
                    
    def save_checkpoint(self, epoch: int, is_best: bool = False, is_regular: bool = False) -> str:
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_name': self.model_name,
            'dataset_name': self.dataset_name,
            'state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'total_loss': self.total_loss_history,
            'best_f1': self.best_f1,
            'config': self.config.__dict__
        }
        
        if is_best:
            # 最佳模型保存在实验根目录
            save_path = self.experiment_dir / "best.pth"
        elif is_regular:
            # 定期检查点保存在checkpoints目录
            save_path = self.checkpoints_dir / f"epoch_{epoch}.pth"
        else:
            # 最新模型保存在实验根目录
            save_path = self.experiment_dir / "latest.pth"
            
        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved to {save_path}")
        return str(save_path)
        
    def log_metrics(self, epoch: int, train_loss: float, val_metrics: Optional[Dict] = None):
        """Log metrics to SwanLab and logger"""
        # Log to SwanLab
        if self.swanlab_run is not None:
            log_data = {
                "epoch": epoch,
                "train_loss": train_loss,
                "learning_rate": self.optimizer.param_groups[0]['lr']
            }
            
            if val_metrics:
                log_data.update({
                    "val_miou": val_metrics.get('mIoU', [0])[-1],
                    "val_iou": val_metrics.get('IoU', 0),
                    "val_niou": val_metrics.get('nIoU', 0),
                    "val_pd": val_metrics.get('PD_FA', [0, 0])[0],
                    "val_fa": val_metrics.get('PD_FA', [0, 0])[1],
                    "val_f1": val_metrics.get('F1', 0),
                    "val_mse": val_metrics.get('MSE', 0),
                })
            
            swanlab.log(log_data)
        
        # Log to file
        if val_metrics:
            logger.info(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, "
                       f"Val mIoU: {val_metrics.get('mIoU', [0])[-1]:.4f}, "
                       f"Val IoU: {val_metrics.get('IoU', 0):.4f}, "
                       f"Val F1: {val_metrics.get('F1', 0):.4f}, "
                       f"Val MSE: {val_metrics.get('MSE', 0):.4f}")
        else:
            logger.info(f"Epoch {epoch} - Train Loss: {train_loss:.4f}")
        
    def compute_orthogonal_loss(self) -> torch.Tensor:
        """Compute orthogonal regularization loss"""
        if not self.config.use_orthogonal_reg:
            return torch.tensor(0.0, device=self.device)
            
        reg_weight = 1e-6
        orth_loss = torch.tensor(0.0, device=self.device)
        
        # Collect parameters by stage
        stage_params = {f'stage{i}': [] for i in range(1, 5)}
        
        for name, param in self.model.named_parameters():
            if 'bias' not in name and 'norm' not in name:
                for stage in stage_params:
                    if f'{stage}.0.branch_conv_list' in name:
                        param_flat = param.view(param.shape[0], -1)
                        stage_params[stage].append(param_flat)
                        break
        
        # Compute orthogonal loss for each stage
        for stage, params in stage_params.items():
            if params:
                param_flat = torch.cat(params, dim=0)
                sym = torch.mm(param_flat, torch.t(param_flat))
                sym -= torch.eye(param_flat.shape[0], device=sym.device)
                orth_loss += reg_weight * sym.abs().sum()
                
        return orth_loss / len(stage_params)
        
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = []
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, (images, targets) in enumerate(progress_bar):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Skip single sample batches
            if images.shape[0] == 1:
                continue
                
            self.optimizer.zero_grad()
            
            # Forward pass with optional AMP
            if self.config.use_amp and self.scaler:
                with torch.cuda.amp.autocast():
                    predictions = self.model(images)
                    loss = self.model.loss(predictions, targets, batch_idx, epoch, images)
                    loss += self.compute_orthogonal_loss()
                    
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(images)
                loss = self.model.loss(predictions, targets, batch_idx, epoch, images)
                loss += self.compute_orthogonal_loss()
                
                loss.backward()
                self.optimizer.step()
            
            # Update EMA
            self.ema.update()
            
            # Record loss
            epoch_losses.append(loss.detach().cpu().item())
            
            # Update progress bar
            avg_loss = np.mean(epoch_losses)
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{current_lr:.6f}'
            })
            
        return np.mean(epoch_losses)
        
    def validate(self, checkpoint_path: str) -> Dict[str, Any]:
        """Validate model and return metrics"""
        # Load model for validation
        val_model = Net(model_name=self.model_name, mode='test').to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        val_model.load_state_dict(checkpoint['state_dict'])
        
        # Switch to deploy mode for specific models
        if self.model_name in ['RepirDet', 'ExtractOne']:
            val_model.model.switch_to_deploy()
            
        val_model.eval()
        
        # Initialize metrics
        metrics = {
            'mIoU': mIoU(),
            'PD_FA': PD_FA(),
            'IoU': SigmoidMetric(),
            'nIoU': SamplewiseSigmoidMetric(1, score_thresh=0.5),
            'F1': F1Metric(threshold=0.5, distance_threshold=5.0),
            'MSE': MSEMetric(threshold=0.5, distance_threshold=5.0)
        }
        
        # Reset metrics
        for metric in metrics.values():
            if hasattr(metric, 'reset'):
                metric.reset()
                
        inference_time = 0.0
        
        with torch.no_grad():
            for images, targets, sizes, _ in self.val_loader:
                images = images.to(self.device)
                
                # Measure inference time
                start_time = time.time()
                predictions = val_model(images)
                end_time = time.time()
                inference_time += (end_time - start_time)
                
                # Handle different prediction formats
                if isinstance(predictions, list):
                    if self.model_name == 'LKUnet':
                        predictions = predictions[-1]
                    else:
                        predictions = predictions[0]
                
                # Apply threshold
                threshold = 0.5
                threshold_tensor = torch.tensor(threshold, device=predictions.device)
                binary_predictions = (predictions > threshold_tensor).float()
                
                # Update metrics
                metrics['mIoU'].update(binary_predictions.cpu(), targets)
                metrics['PD_FA'].update(binary_predictions[0, 0].cpu(), targets[0, 0], sizes)
                metrics['nIoU'].update(binary_predictions.cpu(), targets)
                metrics['IoU'].update(binary_predictions.cpu(), targets)
                metrics['F1'].update(predictions.cpu(), targets)
                metrics['MSE'].update(predictions.cpu(), targets)
        
        # Get results
        results = {
            'mIoU': metrics['mIoU'].get(),
            'PD_FA': metrics['PD_FA'].get(),
            'IoU': metrics['IoU'].get(),
            'nIoU': metrics['nIoU'].get()[-1],
            'F1': metrics['F1'].get(),
            'MSE': metrics['MSE'].get()
        }
        
        # Log results
        logger.info(f"Validation Results:")
        logger.info(f"  mIoU: {results['mIoU']}")
        logger.info(f"  PD, FA: {results['PD_FA']}")
        logger.info(f"  IoU: {results['IoU']}")
        logger.info(f"  nIoU: {results['nIoU']}")
        logger.info(f"  F1: {results['F1']:.4f}")
        logger.info(f"  MSE: {results['MSE']:.4f}")
        logger.info(f"  Inference time: {inference_time:.4f}s")
        
        # Calculate FPS if requested
        if self.config.calculate_fps:
            self._calculate_fps(val_model)
            
        return results
        
    def _calculate_fps(self, model: nn.Module):
        """Calculate FPS for the model"""
        try:
            FPSBenchmark(
                model=model,
                device=str(self.device),
                datasets=self.val_loader,
                iterations=len(self.val_loader),
                log_interval=10
            ).measure_inference_speed()
        except Exception as e:
            logger.warning(f"Failed to calculate FPS: {e}")
            
    def train(self):
        """Main training loop"""
        logger.info(f"Starting training for {self.model_name} on {self.dataset_name}")
        logger.info(f"Training for {self.config.num_epochs} epochs")
        logger.info(f"Save interval: {self.config.save_interval}, Validation interval: {self.config.val_interval}")
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            # Train one epoch
            epoch_loss = self.train_epoch(epoch)
            self.total_loss_history.append(epoch_loss)
            
            # Log metrics
            self.log_metrics(epoch + 1, epoch_loss)
            
            # Save regular checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(epoch + 1, is_regular=True)
            
            # Validate and save best model
            if (epoch + 1) % self.config.val_interval == 0:
                checkpoint_path = self.save_checkpoint(epoch + 1)
                val_metrics = self.validate(checkpoint_path)
                
                # Update best model if improved
                current_f1 = val_metrics['F1']
                if current_f1 > self.best_f1:
                    self.best_f1 = current_f1
                    self.save_checkpoint(epoch + 1, is_best=True)
                    logger.info(f"New best F1: {self.best_f1:.4f} at epoch {epoch+1}")
                
                # Log validation metrics
                self.log_metrics(epoch + 1, epoch_loss, val_metrics)
            
            # Step scheduler
            self.scheduler.step()
            
        # Save final checkpoint
        if (self.config.num_epochs) % self.config.save_interval != 0:
            self.save_checkpoint(self.config.num_epochs, is_regular=True)
            
        # Final validation
        if (self.config.num_epochs) % self.config.val_interval != 0:
            checkpoint_path = self.save_checkpoint(self.config.num_epochs)
            val_metrics = self.validate(checkpoint_path)
            self.log_metrics(self.config.num_epochs, self.total_loss_history[-1], val_metrics)
            
        logger.info(f"Training completed. Best F1: {self.best_f1:.4f}")
        
        # Close SwanLab run
        if self.swanlab_run is not None:
            swanlab.finish()


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Optimized PyTorch Geo-IRSTD Training")
    
    # Model and dataset
    parser.add_argument("--model_names", default=['WTNet'], nargs='+',
                       help="Model names to train")
    parser.add_argument("--dataset_names", default=['spotgeov2-IRSTD'], nargs='+',
                       help="Dataset names to use")
    parser.add_argument("--dataset_dir", default='./datasets', type=str,
                       help="Dataset directory")
    parser.add_argument("--img_norm_cfg", default=None, type=dict,
                       help="Image normalization configuration")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Training batch size")
    parser.add_argument("--patch_size", type=int, default=512,
                       help="Training patch size")
    parser.add_argument("--num_epochs", type=int, default=400,
                       help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Optimization
    parser.add_argument("--optimizer_name", default='Adamw', type=str,
                       help="Optimizer name")
    parser.add_argument("--learning_rate", type=float, default=5e-4,
                       help="Learning rate")
    parser.add_argument("--scheduler_name", default='CosineAnnealingLR', type=str,
                       help="Scheduler name")
    parser.add_argument("--scheduler_settings", default={'epochs': 800, 'min_lr': 0.0005}, type=dict,
                       help="Scheduler settings")
    
    # System
    parser.add_argument("--num_threads", type=int, default=0,
                       help="Number of data loader threads")
    parser.add_argument("--save_dir", default='./log', type=str,
                       help="Save directory")
    parser.add_argument("--resume_paths", default=None, nargs='+',
                       help="Checkpoint paths to resume from")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Prediction threshold")
    parser.add_argument("--use_augmentation", type=bool, default=True,
                       help="Use data augmentation")
    parser.add_argument("--use_orthogonal_reg", type=bool, default=False,
                       help="Use orthogonal regularization")
    parser.add_argument("--calculate_fps", type=bool, default=False,
                       help="Calculate FPS during validation")
    
    # Advanced features
    parser.add_argument("--use_amp", type=bool, default=False,
                       help="Use automatic mixed precision")
    parser.add_argument("--ema_decay", type=float, default=0.0,
                       help="EMA decay rate")
    
    # Monitoring and checkpointing
    parser.add_argument("--save_interval", type=int, default=10,
                       help="Save checkpoint every N epochs")
    parser.add_argument("--val_interval", type=int, default=5,
                       help="Validate every N epochs")
    parser.add_argument("--use_swanlab", type=bool, default=True,
                       help="Enable SwanLab monitoring")
    
    return parser.parse_args()


def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create configuration
    config = TrainingConfig.from_args(args)
    
    # Train for each dataset and model combination
    for dataset_name in config.dataset_names:
        for model_name in config.model_names:
            logger.info(f"Training {model_name} on {dataset_name}")
            
            try:
                trainer = ModelTrainer(config, model_name, dataset_name)
                trainer.train()
            except Exception as e:
                logger.error(f"Training failed for {model_name} on {dataset_name}: {e}")
                continue
                
            logger.info(f"Completed training {model_name} on {dataset_name}")


if __name__ == '__main__':
    main() 