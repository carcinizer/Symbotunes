import time
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_info


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Only reset stats if using a CUDA device
        device = trainer.strategy.root_device
        if not (isinstance(device, torch.device) and device.type == 'cuda'):
            return
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        # Only report if using a CUDA device
        device = trainer.strategy.root_device
        if not (isinstance(device, torch.device) and device.type == 'cuda'):
            return
        torch.cuda.synchronize(device)
        max_memory = torch.cuda.max_memory_allocated(device) / 2**20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.strategy.reduce(max_memory)
            epoch_time = trainer.strategy.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass
