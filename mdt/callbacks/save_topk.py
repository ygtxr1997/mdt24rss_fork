import os
import os.path
import warnings
from typing import Any, Dict, List, Optional
import logging
import heapq
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import STEP_OUTPUT

logger = logging.getLogger(__name__)


class ManuallySaveModelCallback(Callback):
    """
    pl.callbacks.ModelCheckpoint is not working when manually backward, we have to save models here.
    Features:
    - Save top-k checkpoints based on a monitor metric
    - Always save latest.ckpt
    - Handle both 'min' and 'max' modes for monitoring
    """

    def __init__(
            self,
            dirpath: Optional[str] = None,
            filename: Optional[str] = None,
            monitor: Optional[str] = None,
            verbose: bool = False,
            save_last: bool = True,
            save_top_k: int = 1,
            mode: str = "min",
            auto_insert_metric_name: bool = True,
            every_n_epochs: int = 1,
            save_weights_only: bool = False,
            save_on_train_epoch_end: Optional[bool] = None,
    ):
        """
        Args:
            dirpath: Directory to save checkpoints
            filename: Checkpoint filename template. Can include {epoch} and {step} placeholders
            monitor: Metric to monitor for saving best checkpoints
            verbose: Whether to print save messages
            save_last: Whether to always save latest checkpoint
            save_top_k: Number of best checkpoints to keep (based on monitor)
            mode: 'min' or 'max' for monitor metric
            auto_insert_metric_name: Whether to automatically add metric name to filename
            every_n_epochs: Save checkpoint every N epochs
            save_weights_only: Save only model weights (no optimizer states)
            save_on_train_epoch_end: Whether to save on train epoch end
        """
        super().__init__()

        self.dirpath = dirpath or os.getcwd()
        self.filename = filename
        self.monitor = monitor
        self.verbose = verbose
        self.save_last = save_last
        self.save_top_k = save_top_k
        self.mode = mode
        self.auto_insert_metric_name = auto_insert_metric_name
        self.every_n_epochs = every_n_epochs
        self.save_weights_only = save_weights_only
        self.save_on_train_epoch_end = save_on_train_epoch_end

        # Validate mode
        if mode not in ["min", "max"]:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

        # Create directory if it doesn't exist
        os.makedirs(self.dirpath, exist_ok=True)

        # Internal state
        self.best_k_models = []  # heap for top-k models: [(score, epoch, filepath), ...]
        self.kth_best_model_path = ""
        self.best_model_score = float('inf') if mode == 'min' else float('-inf')
        self.best_model_path = ""
        self.last_model_path = ""

        # File format
        if self.filename is None:
            if self.auto_insert_metric_name and self.monitor:
                # 处理斜杠问题，将斜杠替换为下划线
                safe_monitor = self.monitor.replace('/', '_').replace('\\', '_')
                self.filename = f"epoch={{epoch:02d}}-{safe_monitor}={{{self.monitor}:.4f}}"
            else:
                self.filename = "epoch={epoch:02d}"

        if self.verbose:
            rank_zero_info(
                f"ManuallySaveModelCallback initialized:\n"
                f"  dirpath: {self.dirpath}\n"
                f"  monitor: {self.monitor}\n"
                f"  save_top_k: {self.save_top_k}\n"
                f"  mode: {self.mode}\n"
                f"  save_last: {self.save_last}"
            )

    def _get_metric_value(self, trainer: Trainer, pl_module: LightningModule) -> Optional[float]:
        """Get the current value of the monitored metric."""
        if not self.monitor:
            return None

        # Try to get from logged metrics
        if hasattr(trainer, 'logged_metrics') and self.monitor in trainer.logged_metrics:
            return trainer.logged_metrics[self.monitor].item()

        # Try to get from callback metrics
        if hasattr(trainer, 'callback_metrics') and self.monitor in trainer.callback_metrics:
            return trainer.callback_metrics[self.monitor].item()

        # Try to get from progress bar metrics
        if hasattr(trainer, 'progress_bar_metrics') and self.monitor in trainer.progress_bar_metrics:
            return trainer.progress_bar_metrics[self.monitor].item()

        rank_zero_warn(f"Metric '{self.monitor}' not found in trainer metrics.")
        return None

    def _is_better_score(self, current: float, best: float) -> bool:
        """Check if current score is better than best score."""
        if self.mode == "min":
            return current < best
        else:  # mode == "max"
            return current > best

    def _format_checkpoint_name(
            self,
            trainer: Trainer,
            pl_module: LightningModule,
            metrics: Optional[Dict[str, float]] = None
    ) -> str:
        """Format checkpoint filename with current epoch and metrics."""
        filename = self.filename

        # Add .ckpt extension if not present
        if not filename.endswith('.ckpt'):
            filename += '.ckpt'

        try:
            # Format with epoch and step
            filename = filename.format(
                epoch=trainer.current_epoch,
                step=trainer.global_step,
                **(metrics or {})
            )
        except Exception as e:
            # 如果格式化失败，使用简单的文件名
            if self.verbose:
                rank_zero_warn(f"Filename formatting failed: {e}, using fallback")
            filename = f"epoch={trainer.current_epoch:02d}-step={trainer.global_step}.ckpt"

        return filename

    def _save_checkpoint(
            self,
            trainer: Trainer,
            pl_module: LightningModule,
            filepath: str
    ) -> None:
        """Save a checkpoint to the specified filepath."""
        try:
            # Prepare basic checkpoint data
            checkpoint = {
                'epoch': trainer.current_epoch,
                'global_step': trainer.global_step,
                'pytorch-lightning_version': pl.__version__,
                'state_dict': pl_module.state_dict(),
                'lr_schedulers': [],
                'optimizer_states': [],
                'callbacks': {},
                'hyper_parameters': getattr(pl_module, 'hparams', {}),
            }

            # 安全地添加训练循环信息（适配 PyTorch Lightning 1.9.5）
            try:
                if hasattr(trainer, 'fit_loop') and hasattr(trainer.fit_loop, 'epoch_loop'):
                    epoch_loop = trainer.fit_loop.epoch_loop

                    # 添加 batch_idx 信息
                    if hasattr(epoch_loop, 'batch_idx'):
                        checkpoint['epoch_loop.batch_idx'] = epoch_loop.batch_idx

                    # 尝试添加手动优化信息（如果存在）
                    if hasattr(epoch_loop, 'manual_optimization'):
                        manual_opt = epoch_loop.manual_optimization
                        if hasattr(manual_opt, 'optim_step_progress') and hasattr(manual_opt.optim_step_progress,
                                                                                  'total'):
                            checkpoint[
                                'epoch_loop.manual_optimization.optim_step_progress'] = manual_opt.optim_step_progress.total.completed

                    # 添加循环信息
                    loops_info = {
                        'fit_loop': {
                            'epoch_loop.batch_idx': getattr(epoch_loop, 'batch_idx', 0),
                        }
                    }

                    # 添加调度器进度信息（如果存在）
                    if hasattr(epoch_loop, 'scheduler_progress') and hasattr(epoch_loop.scheduler_progress, 'total'):
                        loops_info['fit_loop'][
                            'epoch_loop.scheduler_progress'] = epoch_loop.scheduler_progress.total.completed

                    checkpoint['loops'] = loops_info

            except AttributeError as attr_error:
                # 如果某些属性不存在，继续执行但记录警告
                if self.verbose:
                    rank_zero_warn(f"Could not save some loop information: {attr_error}")

            # Add optimizer states if not save_weights_only
            if not self.save_weights_only:
                try:
                    optimizers = trainer.optimizers
                    if optimizers:
                        if isinstance(optimizers, list):
                            checkpoint['optimizer_states'] = [opt.state_dict() for opt in optimizers]
                        else:
                            checkpoint['optimizer_states'] = [optimizers.state_dict()]
                except Exception as opt_error:
                    if self.verbose:
                        rank_zero_warn(f"Could not save optimizer states: {opt_error}")

                # Add lr scheduler states
                try:
                    lr_schedulers = getattr(trainer, 'lr_scheduler_configs', [])
                    if lr_schedulers:
                        checkpoint['lr_schedulers'] = []
                        for config in lr_schedulers:
                            scheduler_info = {
                                'scheduler': config.scheduler.state_dict(),
                                'interval': config.interval,
                                'frequency': config.frequency,
                            }
                            # 安全地添加 monitor 字段
                            if hasattr(config, 'monitor'):
                                scheduler_info['monitor'] = config.monitor
                            checkpoint['lr_schedulers'].append(scheduler_info)
                except Exception as lr_error:
                    if self.verbose:
                        rank_zero_warn(f"Could not save lr scheduler states: {lr_error}")

            # Add callback states
            try:
                for callback in trainer.callbacks:
                    if hasattr(callback, 'state_dict'):
                        try:
                            callback_state = callback.state_dict()
                            if callback_state:  # 只有非空状态才保存
                                checkpoint['callbacks'][callback.__class__.__name__] = callback_state
                        except Exception as cb_error:
                            if self.verbose:
                                rank_zero_warn(
                                    f"Could not save callback {callback.__class__.__name__} state: {cb_error}")
            except Exception as callbacks_error:
                if self.verbose:
                    rank_zero_warn(f"Could not save callback states: {callbacks_error}")

            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Save checkpoint
            torch.save(checkpoint, filepath)

            if self.verbose:
                rank_zero_info(f"Checkpoint saved: {filepath}")

        except Exception as e:
            rank_zero_warn(f"Failed to save checkpoint to {filepath}: {e}")
            # 添加更详细的错误信息用于调试
            import traceback
            if self.verbose:
                rank_zero_warn(f"Detailed error traceback:\n{traceback.format_exc()}")

    def _save_last_checkpoint(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Save the latest checkpoint."""
        if not self.save_last:
            return

        last_filepath = os.path.join(self.dirpath, "latest.ckpt")
        self._save_checkpoint(trainer, pl_module, last_filepath)
        self.last_model_path = last_filepath

    def _save_top_k_checkpoint(
            self,
            trainer: Trainer,
            pl_module: LightningModule,
            current_score: float
    ) -> None:
        """Save checkpoint if it's among top-k best models."""
        if self.save_top_k <= 0:
            return

        # Format filename with current metrics
        metrics = {self.monitor: current_score} if self.monitor else {}
        filename = self._format_checkpoint_name(trainer, pl_module, metrics)
        filepath = os.path.join(self.dirpath, filename)

        # Update best model info
        if self._is_better_score(current_score, self.best_model_score):
            self.best_model_score = current_score
            self.best_model_path = filepath

        # For unlimited saves (save_top_k = -1)
        if self.save_top_k == -1:
            self._save_checkpoint(trainer, pl_module, filepath)
            return

        # Handle top-k logic
        if len(self.best_k_models) < self.save_top_k:
            # Still have space for more models
            if self.mode == "min":
                heapq.heappush(self.best_k_models, (-current_score, trainer.current_epoch, filepath))
            else:
                heapq.heappush(self.best_k_models, (current_score, trainer.current_epoch, filepath))

            self._save_checkpoint(trainer, pl_module, filepath)

        else:
            # Check if current model is better than the worst in top-k
            if self.mode == "min":
                worst_score = -self.best_k_models[0][0]
                if current_score < worst_score:
                    # Remove worst model
                    _, _, old_filepath = heapq.heappop(self.best_k_models)
                    if os.path.exists(old_filepath):
                        os.remove(old_filepath)
                        if self.verbose:
                            rank_zero_info(f"Removed checkpoint: {old_filepath}")

                    # Add new model
                    heapq.heappush(self.best_k_models, (-current_score, trainer.current_epoch, filepath))
                    self._save_checkpoint(trainer, pl_module, filepath)
            else:
                worst_score = self.best_k_models[0][0]
                if current_score > worst_score:
                    # Remove worst model
                    _, _, old_filepath = heapq.heappop(self.best_k_models)
                    if os.path.exists(old_filepath):
                        os.remove(old_filepath)
                        if self.verbose:
                            rank_zero_info(f"Removed checkpoint: {old_filepath}")

                    # Add new model
                    heapq.heappush(self.best_k_models, (current_score, trainer.current_epoch, filepath))
                    self._save_checkpoint(trainer, pl_module, filepath)

    def _should_save_on_epoch(self, trainer: Trainer) -> bool:
        """Check if we should save checkpoint on this epoch."""
        return trainer.current_epoch % self.every_n_epochs == 0

    @rank_zero_only
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        """Called when the validation epoch ends."""
        if not self._should_save_on_epoch(trainer):
            return

        if self.verbose:
            rank_zero_info('[ManuallySaveModelCallback] Ready to save model')

        # Always save latest checkpoint
        self._save_last_checkpoint(trainer, pl_module)

        # Save top-k checkpoint if monitor is specified
        if self.monitor:
            current_score = self._get_metric_value(trainer, pl_module)
            if current_score is not None:
                self._save_top_k_checkpoint(trainer, pl_module, current_score)

                if self.verbose:
                    rank_zero_info(
                        f"Epoch {trainer.current_epoch}: {self.monitor}={current_score:.4f}, "
                        f"Best: {self.best_model_score:.4f}"
                    )
            else:
                rank_zero_warn(f"Monitor metric '{self.monitor}' not available for epoch {trainer.current_epoch}")

    @rank_zero_only
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        """Called when the training epoch ends."""
        if self.save_on_train_epoch_end and self._should_save_on_epoch(trainer):
            self.on_validation_epoch_end(trainer, pl_module)

    def state_dict(self) -> Dict[str, Any]:
        """Return the callback state."""
        return {
            'best_k_models': self.best_k_models,
            'kth_best_model_path': self.kth_best_model_path,
            'best_model_score': self.best_model_score,
            'best_model_path': self.best_model_path,
            'last_model_path': self.last_model_path,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load the callback state."""
        self.best_k_models = state_dict.get('best_k_models', [])
        self.kth_best_model_path = state_dict.get('kth_best_model_path', "")
        self.best_model_score = state_dict.get('best_model_score',
                                               float('inf') if self.mode == 'min' else float('-inf'))
        self.best_model_path = state_dict.get('best_model_path', "")
        self.last_model_path = state_dict.get('last_model_path', "")

    @property
    def best_model_paths(self) -> List[str]:
        """Return paths of all top-k models."""
        return [path for _, _, path in self.best_k_models]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"dirpath={self.dirpath}, "
            f"monitor={self.monitor}, "
            f"save_top_k={self.save_top_k}, "
            f"mode={self.mode}, "
            f"save_last={self.save_last})"
        )