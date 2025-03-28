"""
This module implements tools functions to manage external models used within the component
"""

import hashlib
import json
import logging
import os.path
from pathlib import Path
from .twe_logger import log_and_raise
import urllib.request
import yaml

# This variable shoud point to python env root_path
ROOT_PATH = Path(__file__).parent.parent.parent.resolve()


class ModelManager:
    """
    Manages all external models required by the library
    """

    def __init__(self, enhancer: str, required_model: str, logger: logging.Logger):
        self._logger = logger
        self._checksums = self._load_checksums()
        self._enhancer = enhancer
        self._model_filename = required_model
        self._enhancer_directory = (
            Path(os.path.expanduser("~")) / ".neuralde" / enhancer
        )
        self._model_filepath = self._enhancer_directory / required_model

        # Load model repostories
        with open(
            ROOT_PATH / "neural_de/external/_repositories/external_models_list.yaml",
            encoding="utf-8",
        ) as stream:
            try:
                self.external_models_list = yaml.safe_load(stream)
            except yaml.YAMLError:
                print(
                    "error the external models list file is missing from package , it should be at \
                         /neural_de/external/_repositories/external_models_list.yaml"
                )

    @staticmethod
    def _load_checksums():
        """
        Load each available model's checksum.
        """
        with open(
            ROOT_PATH / "neural_de/external/_checksums/checksums.json",
            "r",
            encoding="utf-8",
        ) as checksum_file:
            checksums = json.load(checksum_file)
        return checksums

    def download_model(self) -> None:
        """
        Download weights for an enhancer if they are not already available locally.
        Weights will be stored at ~/.neuralde/{enhancer_name}/model.pth
        """
        if not (self._is_model_available() and self._is_model_valid()):
            self._logger.info(
                "Model %s not found locally or corrupted, downloading it from server",
                self._model_filename,
            )

            if self._enhancer in self.external_models_list.keys():
                print(
                    "Required pretrain model for ",
                    self._enhancer,
                    " enchancer is not present in local cache, downloading it . . .",
                )
                urllib.request.urlretrieve(
                    self.external_models_list[self._enhancer], self._model_filepath
                )
                self._check_download_status()
                print("Pretrained model has been downloaded in cache")
            else:
                raise ValueError(
                    "error there is no defined model repository for the enhancer ",
                    self._enhancer,
                    "in external models annuary",
                )
        else:
            self._logger.info("Model already available locally, skipping download")

    def _check_download_status(self):
        """
        Validate if the model is locally present and valid, and raises an error if not.
        """
        if not self._is_model_valid():
            self._remove_corrupted_model()
            log_and_raise(
                self._logger,
                ValueError,
                "The downloaded file does not pass the checksum validation,"
                " it might be invalid. It has been removed from your machine",
            )
        else:
            self._logger.info("Model downloaded and validated")

    def _remove_corrupted_model(self):
        if self._model_filepath.is_file():
            os.remove(self._model_filepath)
        else:
            log_and_raise(
                self._logger, FileNotFoundError, "Expected model was not found"
            )

    def _is_model_available(self) -> bool:
        self._enhancer_directory.mkdir(parents=True, exist_ok=True)
        if self._model_filepath.is_file():
            return True
        return False

    def _is_model_valid(self) -> bool:
        return self._calculate_checksum() == self._checksums[self._model_filename]

    def _calculate_checksum(self):
        hash_md5 = hashlib.md5()
        if self._model_filepath.is_file():
            with open(self._model_filepath, "rb") as file:
                for chunk in iter(lambda: file.read(4096), b""):
                    hash_md5.update(chunk)
        else:
            log_and_raise(
                self._logger, FileNotFoundError, "Expected model was not found"
            )
        return hash_md5.hexdigest()
