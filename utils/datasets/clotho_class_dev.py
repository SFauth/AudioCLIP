import os
import warnings
import multiprocessing as mp

import tqdm
import librosa
import soundfile as sf

import numpy as np
import pandas as pd

import torch.utils.data as td

import sklearn.model_selection as skms

sys.path.insert(0, '/shared-network/sfauth/data/AudioCLIP') # overcomes transform.py being in another directory

import utils.transforms as transforms

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Optional

# Clotho class

class Clotho(td.Dataset):


    def __init__(self,
                 root: str,
                 sample_rate: int = 22050,
                 mono: bool = False,
                 n_cpus_to_not_use=0,
                 transform_audio=None,
                 target_transform=None,
                 **_):

        super(Clotho, self).__init__()

        self.root = root
        self.sample_rate = sample_rate
        self.mono = mono
        self.transform = transform_audio # transforms input? 
        self.target_transform = target_transform # transforms output?
        self.n_cpus_to_not_use = n_cpus_to_not_use

    
    @staticmethod
    def _load_worker(fn: str, path_to_file: str, sample_rate: int, mono: bool = False) -> Tuple[str, int, np.ndarray]:
        wav, sample_rate_ = sf.read(  #reads in sound data
            path_to_file,
            dtype='float32',
            always_2d=True
        )
        wav = librosa.resample(wav.T, sample_rate_, sample_rate)  #converts the sample rate to the specified sample rate

        if wav.shape[0] == 1 and not mono:
            wav = np.concatenate((wav, wav), axis=0)

        wav = wav[:, :sample_rate * 4]
        wav = transforms.scale(wav, wav.min(), wav.max(), -32768.0, 32767.0) # normalizing to -32768 until 32767

        return fn, sample_rate, wav.astype(np.float32)

    def load_data(self):

        # read in metadata 

        meta = pd.read_csv(
            os.path.join(root, 'metadata', 'metadata.csv'),
            sep=',',
            index_col='file_name'.assign(keywords = lambda df: df["keywords"].map(lambda keywords: keywords.split(";")))

        )
        # keep only first caption

        meta = meta.assign(keywords = meta.keywords.map(lambda x: x[0]))

        
        for row_idx, (fn, row) in enumerate(meta.iterrows()):
            path = os.path.join(root, "test_data", fn)
            data[fn] = path, sample_rate, mono

        self.indices = {idx: fn for idx, fn in enumerate(self.data)}

        num_processes = os.cpu_count() - n_cpus_to_not_use # leave 42 CPUs free

        warnings.filterwarnings('ignore')
        with mp.Pool(processes=num_processes) as pool:
            tqdm.tqdm.write(f'Loading test data')

            for fn, sample_rate, wav in pool.starmap(func=_load_worker,
                        iterable=[(fn, path, sample_rate, mono) for fn, (path, sample_rate, mono) in data.items()],
                        chunksize=int(np.ceil(len(meta) / num_processes)) or 1):
                    # iterable is dict in original code. fold 1 : path, sample_rate, mono

                    data[fn] = {
                    'audio': wav,
                    'sample_rate': sample_rate,
                    'target': meta.loc[fn]["keywords"],
                    'sound_id': meta.loc[fn]['sound_id'],
                    'sound_link': meta.loc[fn]['sound_link'],
                    'start_end_samples' :  meta.loc[fn]['start_end_samples'],
                    'manufacturer' : meta.loc[fn]['manufacturer'],
                    'license' : meta.loc[fn]['license']
                }
                


        def __getitem__(self, index: int) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
            if not (0 <= index < len(self)):
                raise IndexError

            audio: np.ndarray = self.data[self.indices[index]]['audio']
            target: str = self.data[self.indices[index]]['target']

            if self.transform is not None:
                audio = self.transform(audio)
            if self.target_transform is not None:
                target = self.target_transform(target)

        return audio, None, [target]



    def __len__(self) -> int:
        return len(self.data)

# code for class

root = '/shared-network/sfauth/data/AudioCLIP/data/test'
mono = False
sample_rate = int(22050)
n_cpus_to_not_use = 42
data = dict()

# load metadata

meta = pd.read_csv(
            os.path.join(root, 'metadata', 'metadata.csv'),
            sep=',',
            index_col='file_name'
        ).assign(keywords = lambda df: df["keywords"].map(lambda keywords: keywords.split(";")))

# load captions 

captions = pd.read_csv(
            os.path.join(root, 'metadata', 'captions.csv'),
            sep=',',
            index_col='file_name'
        )
        

def _load_worker(fn: str, path_to_file: str, sample_rate: int, mono: bool = False) -> Tuple[str, int, np.ndarray]:
        wav, sample_rate_ = sf.read(  #reads in sound data
            path_to_file,
            dtype='float32',
            always_2d=True
        )
        wav = librosa.resample(wav.T, sample_rate_, sample_rate)  #converts the sample rate to the specified sample rate

        if wav.shape[0] == 1 and not mono:
            wav = np.concatenate((wav, wav), axis=0)

        wav = wav[:, :sample_rate * 4]
        wav = transforms.scale(wav, wav.min(), wav.max(), -32768.0, 32767.0) # normalizing to -32768 until 32767

        return fn, sample_rate, wav.astype(np.float32)



for row_idx, (fn, row) in enumerate(meta.iterrows()):
        path = os.path.join(root, "test_data", fn)
        data[fn] = path, sample_rate, mono


indices = {idx: fn for idx, fn in enumerate(data)}


num_processes = os.cpu_count() - n_cpus_to_not_use # leave 42 CPUs free
warnings.filterwarnings('ignore')
with mp.Pool(processes=num_processes) as pool:
    tqdm.tqdm.write(f'Loading test data')

    for fn, sample_rate, wav in pool.starmap(func=_load_worker,
                iterable=[(fn, path, sample_rate, mono) for fn, (path, sample_rate, mono) in data.items()],
                chunksize=int(np.ceil(len(meta) / num_processes)) or 1):
            # iterable is dict in original code. fold 1 : path, sample_rate, mono

            data[fn] = {
            'audio': wav,
            'sample_rate': sample_rate,
            'target': meta.loc[fn]["keywords"],
            'sound_id': meta.loc[fn]['sound_id'],
            'sound_link': meta.loc[fn]['sound_link'],
            'start_end_samples' :  meta.loc[fn]['start_end_samples'],
            'manufacturer' : meta.loc[fn]['manufacturer'],
            'license' : meta.loc[fn]['license']
        }


# class

class UrbanSound8K(td.Dataset): # td.Dataset is a blueprint class for a dataset

    def __init__(self,
                 root: str,
                 sample_rate: int = 22050,
                 train: bool = True,
                 fold: Optional[int] = None,
                 mono: bool = False,
                 transform_audio=None,
                 target_transform=None,
                 **_):

        super(UrbanSound8K, self).__init__()  

        r"""initiate blueprint class to get basic functions like __getitem__  (integer indexing to get a sample) 
            and __len__ (shows length of dataset)"""

        self.root = root
        self.sample_rate = sample_rate
        self.train = train
        self.random_split_seed = None

        if fold is None:
            fold = 1

        if not (1 <= fold <= 10):
            raise ValueError(f'Expected fold in range [1, 10], got {fold}')

        self.fold = fold  # training folds
        self.folds_to_load = set(range(1, 11)) # converts the folds to be iterable

        if self.fold not in self.folds_to_load:
            raise ValueError(f'fold {fold} does not exist')

        if self.train:
            # if in training mode, keep all but test fold
            self.folds_to_load -= {self.fold}
        else:
            # if in evaluation mode, keep the test samples only
            self.folds_to_load -= self.folds_to_load - {self.fold}

        self.mono = mono

        self.transform = transform_audio
        self.target_transform = target_transform

        r"""the attribute data should be a dict,
        whose key is a string and whose value is another dict holding a string
        as key and any type as a value.
        Then data is right away defined to be an empty dict"""
        self.data: Dict[str, Dict[str, Any]] = dict() 
        
        self.indices = dict()
        self.load_data()

        self.class_idx_to_label = dict() # create dict that assigns to every file name a label (y_true)
        for row in self.data.values():
            idx = row['target']
            label = row['category']
            self.class_idx_to_label[idx] = label
        self.label_to_class_idx = {lb: idx for idx, lb in self.class_idx_to_label.items()}

    @staticmethod 
    def _load_worker(fn: str, path_to_file: str, sample_rate: int, mono: bool = False) -> Tuple[str, int, np.ndarray]:
        wav, sample_rate_ = sf.read(  #reads in sound data
            path_to_file,
            dtype='float32',
            always_2d=True
        )
        r"""
        staticmethod: Allows to call this function without defining/creating an object (instance of the class)
        _load_worker: if we import all methods from this python script, this function does not get imported
        """
        r"""
        sample rate: e.g 10000 features/second, resulting in 10000 columns
        if our file is 5 seconds of sound: we have 5 seconds * 10000 features/second = 50000 samples (rows)
        """
        wav = librosa.resample(wav.T, sample_rate_, sample_rate)  #converts the sample rate to the specified sample rate

        if wav.shape[0] == 1 and not mono:
            wav = np.concatenate((wav, wav), axis=0)

        wav = wav[:, :sample_rate * 4]
        wav = transforms.scale(wav, wav.min(), wav.max(), -32768.0, 32767.0)

        return fn, sample_rate, wav.astype(np.float32)

    def load_data(self):
        # read metadata
               
        meta = pd.read_csv(
            # data/test/metadata/UrbanSound8K.csv
            os.path.join(self.root, 'metadata', 'UrbanSound8K.csv'),
            sep=',',
            index_col='slice_file_name'
        )

        for row_idx, (fn, row) in enumerate(meta.iterrows()):  #fn: fold number
            r"""
            replaces curly brackets with row['fold'], i.e. fold number
            path = data/test/audio/fold2/2
            """ 
            path = os.path.join(self.root, 'audio', 'fold{}'.format(row['fold']), fn) 
            
            self.data[fn] = path, self.sample_rate, self.mono

        # by default, the official split from the metadata is used
        files_to_load = list()
        # if the random seed is not None, the random split is used
        if self.random_split_seed is not None:
            # given an integer random seed
            skf = skms.StratifiedKFold(n_splits=10, shuffle=True, random_state=self.random_split_seed)

            # split the US8K samples into 10 folds
            for fold_idx, (train_ids, test_ids) in enumerate(skf.split(
                    np.zeros(len(meta)), meta['classID'].values.astype(int)
            ), 1):
                # if this is the fold we want to load, add the corresponding files to the list
                if fold_idx == self.fold:
                    ids = train_ids if self.train else test_ids
                    filenames = meta.iloc[ids].index
                    files_to_load.extend(filenames)
                    break
        else:
            # if the random seed is None, use the official split
            for fn, row in meta.iterrows():
                if int(row['fold']) in self.folds_to_load:
                    files_to_load.append(fn)

        self.data = {fn: vals for fn, vals in self.data.items() if fn in files_to_load}
        self.indices = {idx: fn for idx, fn in enumerate(self.data)}

        num_processes = os.cpu_count()
        warnings.filterwarnings('ignore')
        with mp.Pool(processes=num_processes) as pool:
            tqdm.tqdm.write(f'Loading {self.__class__.__name__} (train={self.train})')
            for fn, sample_rate, wav in pool.starmap(
                func=self._load_worker,
                iterable=[(fn, path, sr, mono) for fn, (path, sr, mono) in self.data.items()],
                chunksize=int(np.ceil(len(meta) / num_processes)) or 1
            ):
                self.data[fn] = {
                    'audio': wav,
                    'sample_rate': sample_rate,
                    'target': meta.loc[fn, 'classID'],
                    'category': meta.loc[fn, 'class'].replace('_', ' ').strip(' '),
                    'background': bool(meta.loc[fn, 'salience'] - 1)
                }

    def __getitem__(self, index: int) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
        if not (0 <= index < len(self)):
            raise IndexError

        audio: np.ndarray = self.data[self.indices[index]]['audio']
        target: str = self.data[self.indices[index]]['category']

        if self.transform is not None:
            audio = self.transform(audio)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return audio, None, [target]

    def __len__(self) -> int:
        return len(self.data)



loader = UrbanSound8K(root="data/test",
                      sample_rate=44100,
                      mono=False)
