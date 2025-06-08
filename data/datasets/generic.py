import os
from typing import Callable

from .base import BaseDataset


class GenericDataset(BaseDataset):
    """
    A generic dataset class for supplying custom local datasets
    """

    def __init__(
        self,
        dirname: str,
        root: str = "_data",
        split: str = "train",
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        preload: bool = True,
        download: bool = True,
        replace_if_exists: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(root, split, download, replace_if_exists, transform, target_transform, **kwargs)
        self.dirname = dirname

        if preload:
            self._load_data()

    def _load_midi_paths(self, directory_path):
        file_list = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                full_path = os.path.join(root, file)
                file_list.append(full_path)
        return file_list

    def _load_data(self):
        self.data = self._load_midi_paths(os.path.join(self.root, "train", self.dirname))

    def download(self) -> None:
        pass
