
from frechet_music_distance import FrechetMusicDistance
import pytorch_lightning as pl
from omegaconf import OmegaConf

from tempfile import TemporaryDirectory
from pathlib import Path



class FrechetMusicDistanceCallback(pl.Callback):
    
    def __init__(
        self, 
        nowname: str,
        config: OmegaConf,
        lightning_config: OmegaConf,
        dl_config: OmegaConf,

        dataset_path: str, 
        batch_size: int, 
        every_n_epochs: int = 5,
        feature_extractor = "clamp2",
        gaussian_estimator = "mle",
    ) -> None:
        super().__init__()

        self.every_n_epochs = every_n_epochs
        self.batch_size = batch_size
        self.dataset_path = dataset_path

        self.fmd = FrechetMusicDistance(feature_extractor, gaussian_estimator)

        self.ref_features = self.fmd._feature_extractor.extract_features(dataset_path)
        self.mean_ref, self.cov_ref = self.fmd._gaussian_estimator.estimate_parameters(self.ref_features)

        self.epoch = 0


    def on_train_epoch_end(self, trainer, pl_module):
        self.epoch += 1
        if self.epoch % self.every_n_epochs != 0:
            return

        midi_contents = pl_module.sample_midi(self.batch_size)

        with TemporaryDirectory() as tmpdir:
            for idx, sample in enumerate(midi_contents):
                path = Path(tmpdir) / f"{idx}.mid"
                path.write_bytes(sample)

            test_features = self.fmd._feature_extractor.extract_features(tmpdir)

        mean_test, cov_test = self.fmd._gaussian_estimator.estimate_parameters(test_features)
        
        mean_test, cov_test = test_features.flatten(), self.cov_ref
        self.mean_ref, self.cov_ref = self.fmd._gaussian_estimator.estimate_parameters(self.ref_features)
        pl_module.log("train/fmd", self.fmd._compute_fmd(self.mean_ref, mean_test, self.cov_ref, cov_test))
