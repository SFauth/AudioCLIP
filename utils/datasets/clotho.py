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
        self.train = train
        self.random_split_seed = None
        self.mono = mono
        self.transform = transform_audio # transforms input? 
        self.target_transform = target_transform # transforms output?
        self.n_cpus_to_not_use = n_cpus_to_not_use
        self.data = dict()

    
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
            index_col='file_name').assign(keywords = lambda df: df["keywords"].map(lambda keywords: keywords.split(";")))

        
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