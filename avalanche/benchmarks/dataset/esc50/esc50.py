from benchmarks.dataset.esc50 import esc50_data
from torch.utils.data import Dataset
from torch.nn import Module
import os
import numpy as np
import librosa
from typing import Tuple
from torch import Tensor
from torchvision.datasets.utils import download_url
from torchaudio.datasets.utils import extract_archive
import torchaudio


# Download URL and checksums
URL = {
    "esc-50": esc50_data.name[1]
    # "esc-us": None,
}

_CHECKSUMS = {
    "esc-50": None,
    # "esc-us": None,
}

def add_white_noise(x, rate=0.002):
        return x + rate*np.random.randn(len(x))

    # data augmentation: shift sound in timeframe
def shift_sound(x, rate=2):
    return np.roll(x, int(len(x)//rate))

    # data augmentation: stretch sound
def stretch_sound(x, rate=1.1):
    input_length = len(x)
    x = librosa.effects.time_stretch(x, rate)
    if len(x)>input_length:
        return x[:input_length]

# Constant
ARCHIVE_BASENAME = esc50_data.name[0]
FOLDER_IN_ARCHIVE = esc50_data.name[0]
AUDIO_FOLDER = "audio"
META_FOLDER = "meta"
AVAILABLE_VERSION = list(URL.keys())

# Default parameters
FOLDS = (1, 2, 3, 4, 5)


class ESC50(Dataset):
    """
    ESC datasets
    Args:
        root (string): Root directory of datasets where directory
            ``ESC-50-master`` exists or will be saved to if download is set to True.
        download (bool, optional): If true, download the dataset from the internet
            and puts it in root directory. If datasets is already downloaded, it is
            not downloaded again.
    """
    NB_CLASS = 50

    def __init__(self,
                 root: str,
                 train: bool = True,
                 folds: tuple = FOLDS,
                  data_aug=False,
                 download: bool = False,
                 transform: Module = None) -> None:

        super().__init__()
        self.train = train
        self.root = root
        self.required_folds = folds
        self.transform = transform
        self.data_aug = data_aug
        self.url = URL["esc-50"]
        self.nb_class = 50
        self.target_directory = os.path.join(self.root, FOLDER_IN_ARCHIVE)

        # Dataset must exist to continue
        if download:
            print('start download')
            self.download()
        # elif not self.check_integrity(self.target_directory):
        #     raise RuntimeError("Dataset not found or corrupted. \n\
        #         You can use download=True to download it.")

        # Prepare the medata
      
        self._filenames = []
        self._folds = []
        self.targets = []
        self._esc10s = []
        self._load_metadata()
    
    def add_white_noise(x, rate=0.002):
        return x + rate*np.random.randn(len(x))

    # data augmentation: shift sound in timeframe
    def shift_sound(x, rate=2):
        return np.roll(x, int(len(x)//rate))

    # data augmentation: stretch sound
    def stretch_sound(x, rate=1.1):
        input_length = len(x)
        x = librosa.effects.time_stretch(x, rate)
        if len(x)>input_length:
            return x[:input_length]
    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (raw_audio, sr, target).
        """
        
        
        data, target = self.load_item(index)
        # augumentations in wave domain.
        
        if self.data_aug:
            r = np.random.rand()
            
            if r < 0.3:
                
                data = add_white_noise(data)
           
            r = np.random.rand()
            if r < 0.3:
               
                data = shift_sound(data, rate=1+np.random.rand())
            
            #r = np.random.rand()
            #if r < 0.3:
                #data = stretch_sound(data, rate=0.8+np.random.rand()*0.4)
        
        if self.transform is not None:
            data = self.transform(data)
            data = data.squeeze()
        
        return np.float32(data), target

    def __len__(self) -> int:
        return len(self._filenames)

    def _load_metadata(self) -> None:
        
        """Read the metadata csv file and gather the information needed."""
        # HEADER COLUMN NUMBER
        c_filename = 0
        c_fold = 1
        c_target = 2
        c_esc10 = 4

        # Read the csv file and remove header
        path = os.path.join(self.target_directory, META_FOLDER, "esc50.csv")
        
        with open(path, "r") as fp:
            data = fp.read().splitlines()[1:]
            
            for line in data:
                items = line.split(",")
                if self.train:
                  if int(items[c_fold]) < 5:
                    self._filenames.append(items[c_filename])
                    self._folds.append(int(items[c_fold]))
                    self.targets.append(int(items[c_target]))
                    self._esc10s.append(eval(items[c_esc10]))
                else:
                  if int(items[c_fold]) == 5:
                    self._filenames.append(items[c_filename])
                    self._folds.append(int(items[c_fold]))
                    self.targets.append(int(items[c_target]))
                    self._esc10s.append(eval(items[c_esc10]))

        self._filenames = np.asarray(self._filenames)
        self._folds = np.asarray(self._folds)
        self.targets = np.asarray(self.targets)
        self._esc10s = np.asarray(self._esc10s)

        # Keep only the required folds
        

        
        folds_mask = sum([self._folds == f for f in self.required_folds]) >= 1
        
        self._filenames = self._filenames[folds_mask]
        self.targets = self.targets[folds_mask]
        self._esc10s = self._esc10s[folds_mask]

    def download(self) -> None:
        
        """Download the dataset and extract the archive"""
        if self.check_integrity(self.target_directory):
            
            print("Dataset already downloaded and verified.")

        else:
           
            archive_path = os.path.join(self.root, FOLDER_IN_ARCHIVE + ".zip")
           
            download_url(self.url, self.root, filename=FOLDER_IN_ARCHIVE + ".zip")
            
            extract_archive(archive_path, self.root)
    # data augmentation: add white noise
    
    def check_integrity(self, path, checksum=None) -> bool:
        
        """Check if the dataset already exist and if yes, if it is not corrupted.
        Returns:
            bool: False if the dataset doesn't exist or if it is corrupted.
        """
        if not os.path.isdir(path):
            return False

        # TODO add checksum verification
        return True

    def load_item(self, index: int) -> Tuple[Tensor, int]:
        
        filename = self._filenames[index]
        target = self.targets[index]

        path = os.path.join(self.target_directory, AUDIO_FOLDER, filename)
        waveform, sample_rate = librosa.load(path, sr=44100)
        
        # waveform, sample_rate = torchaudio.load(path)
        
        return waveform, target


class ESC50_v2(Dataset):
    """
    ESC datasets
    Args:
        root (string): Root directory of datasets where directory
            ``ESC-50-master`` exists or will be saved to if download is set to True.
        download (bool, optional): If true, download the dataset from the internet
            and puts it in root directory. If datasets is already downloaded, it is
            not downloaded again.
    """
    NB_CLASS = 50

    def __init__(self,
                 root: str,
                 train: bool = True,
                 folds: tuple = FOLDS,
                  data_aug=False,
                 download: bool = False,
                 transform: Module = None) -> None:

        super().__init__()
        self.train = train
        self.root = root
        self.required_folds = folds
        self.transform = transform
        self.data_aug = data_aug
        self.url = URL["esc-50"]
        self.nb_class = 50
        self.target_directory = os.path.join(self.root, FOLDER_IN_ARCHIVE)

        # Dataset must exist to continue
        if download:
            print('start download')
            self.download()
        # elif not self.check_integrity(self.target_directory):
        #     raise RuntimeError("Dataset not found or corrupted. \n\
        #         You can use download=True to download it.")

        # Prepare the medata
        
        self._filenames = []
        self._folds = []
        self.targets = []
        self._esc10s = []
        self._load_metadata()
    
    def add_white_noise(x, rate=0.002):
        return x + rate*np.random.randn(len(x))

    # data augmentation: shift sound in timeframe
    def shift_sound(x, rate=2):
        return np.roll(x, int(len(x)//rate))

    # data augmentation: stretch sound
    def stretch_sound(x, rate=1.1):
        input_length = len(x)
        x = librosa.effects.time_stretch(x, rate)
        if len(x)>input_length:
            return x[:input_length]
    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (raw_audio, sr, target).
        """
        
        
        data, target = self.load_item(index)
        # augumentations in wave domain.
        
        if self.data_aug:
            r = np.random.rand()
            
            if r < 0.3:
                
                data = add_white_noise(data)
           
            r = np.random.rand()
            if r < 0.3:
               
                data = shift_sound(data, rate=1+np.random.rand())
            
            #r = np.random.rand()
            #if r < 0.3:
                #data = stretch_sound(data, rate=0.8+np.random.rand()*0.4)
        
        if self.transform is not None:
            data = self.transform(data)
            data = data.squeeze()
        
        return np.float32(data), target

    def __len__(self) -> int:
        return len(self._filenames)

    def _load_metadata(self) -> None:
        
        """Read the metadata csv file and gather the information needed."""
        # HEADER COLUMN NUMBER
        c_filename = 0
        c_fold = 1
        c_target = 2
        c_esc10 = 4

        # Read the csv file and remove header
        path = os.path.join(self.target_directory, META_FOLDER, "esc50.csv")
        
        with open(path, "r") as fp:
            data = fp.read().splitlines()[1:]
            
            for line in data:
                items = line.split(",")
                if self.train:
                  if int(items[c_fold]) < 5:
                    self._filenames.append(items[c_filename])
                    self._folds.append(int(items[c_fold]))
                    self.targets.append(int(items[c_target]))
                    self._esc10s.append(eval(items[c_esc10]))
                else:
                  if int(items[c_fold]) == 5:
                    self._filenames.append(items[c_filename])
                    self._folds.append(int(items[c_fold]))
                    self.targets.append(int(items[c_target]))
                    self._esc10s.append(eval(items[c_esc10]))

        self._filenames = np.asarray(self._filenames)
        self._folds = np.asarray(self._folds)
        self.targets = np.asarray(self.targets)
        self._esc10s = np.asarray(self._esc10s)

        # Keep only the required folds
        

        
        folds_mask = sum([self._folds == f for f in self.required_folds]) >= 1
        
        self._filenames = self._filenames[folds_mask]
        self.targets = self.targets[folds_mask]
        self._esc10s = self._esc10s[folds_mask]
        self.resample = torchaudio.transforms.Resample(
            orig_freq=44100, new_freq=8000
        )
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=8000)
        self.db = torchaudio.transforms.AmplitudeToDB(top_db=80)

    def download(self) -> None:
        
        """Download the dataset and extract the archive"""
        if self.check_integrity(self.target_directory):
            
            print("Dataset already downloaded and verified.")

        else:
            
            archive_path = os.path.join(self.root, FOLDER_IN_ARCHIVE + ".zip")
            
            download_url(self.url, self.root, filename='ESC-50-master.zip')
            
            extract_archive(archive_path, self.root)
    # data augmentation: add white noise
    
    def check_integrity(self, path, checksum=None) -> bool:
        
        """Check if the dataset already exist and if yes, if it is not corrupted.
        Returns:
            bool: False if the dataset doesn't exist or if it is corrupted.
        """
        if not os.path.isdir(path):
            return False

        # TODO add checksum verification
        return True

    def load_item(self, index: int) -> Tuple[Tensor, int]:
        
        filename = self._filenames[index]
        target = self.targets[index]

        path = os.path.join(self.target_directory, AUDIO_FOLDER, filename)
        wav, _ = torchaudio.load(path)
        xb = self.db(
            self.melspec(
                self.resample(wav)
            )
        )
        
        # waveform, sample_rate = torchaudio.load(path)
        
        return xb, target