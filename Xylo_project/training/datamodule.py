
import os
import sys
from pathlib import Path
import logging
import numpy as np
import pytorch_lightning as pl
from tonic import DiskCachedDataset
from typing import Callable, Iterable, Optional, Tuple, Union
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.transforms import Compose
repo_path = str(Path(__file__).absolute().parent.parent)
sys.path.append(repo_path)

# Import dataset
from data.dataset import DATA_PATH, SpokenDigits
from training_utils.cache_utils import create_config_str


class SpokenDigitsDataModule(pl.LightningDataModule):
    """Load and preprocess the ``SpokenDigits``. Create data loaders for
    train/val/test, and applying optional transforms to the data.

    Args:
        batch_size_train (int): The number of data samples bundled up in one mini batch from the train partition.
        batch_size_val (int): The number of data samples bundled up in one mini batch from the val partition.
        batch_size_test (int): The number of data samples bundled up in one mini batch from the test partition.
        batch_class_distribution (list): percentages indicating the probability of each dataset class appearing in the batch 
        params_dataset (int, optional): dataset parameters
        num_workers (int, optional): The number of threads for the dataloader. Defaults to 4.
        prefetch_factor (int, optional): The number of batches to preload.
            NOTE: If num_workers is set to 0, thixs has to be equal to 2. Defaults to 2.
        cache_dataset (bool, optional): save the cache or not (=True recommended). Defaults to True.
        reset_cache (bool, optional): reset the cache saved to the disk or not. Defaults to False.
        cache_path (Optional[str], optional): optional path for storing cache. By default cache is stored under the data package directory. Defaults to None.
        transform (Optional[Iterable[Callable]], optional): An optional list of callables that will be applied to each data sample. Defaults to None.
        target_transform (Optional[Iterable[Callable]], optional): An optional list of callables that will be applied to each label. Defaults to None.
        auto_balance_train_set (bool, optional): When you have an imbalanced classes after slicing the raw events, setting this as True will add a ``WeightedRandomSampler`` to the train dataloader which can solve the class im-balancing.. Defaults to True.
        auto_balance_val_set (bool, optional): When you have an imbalanced classes after slicing the raw events, setting this as True will add a ``WeightedRandomSampler`` to the val dataloader which can solve the class im-balancing... Defaults to False.
        auto_balance_test_set (bool, optional): When you have an imbalanced classes after slicing the raw events, setting this as True will add a ``WeightedRandomSampler`` to the test dataloader which can solve the class im-balancing... Defaults to False.
    """

    def __init__(
        self,
        batch_size_train: int,
        batch_size_val: int,
        batch_size_test: int,
        batch_class_distribution: list=[],
        params_dataset: dict = {},
        num_workers: int = 1,
        prefetch_factor: int = 2,
        cache_dataset: bool = True,
        reset_cache: bool = False,
        cache_path: Optional[str] = None,
        transform: Optional[Iterable[Callable]] = None,
        target_transform: Optional[Iterable[Callable]] = None,
        auto_balance_train_set: bool = False,
        auto_balance_val_set: bool = False,
        auto_balance_test_set: bool = False,
        direct_cached_data: Optional[str] = None,
        
    ) -> None:
        super().__init__()
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.batch_size_test = batch_size_test
        self.params_dataset = params_dataset
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.cache_dataset = cache_dataset
        self.reset_cache = reset_cache
        self.cache_path = (
            os.path.dirname(DATA_PATH) if cache_path is None else cache_path
        )
        self.batch_class_distribution = np.array(batch_class_distribution)
        self.direct_cached_data = direct_cached_data

        # - Transform
        transform = [] if transform is None else [*transform]
        target_transform = [] if target_transform is None else [*target_transform]

        logging.info(
            "Transforms applied:"
            + f"\nTransform\n\t{transform}"
            + f"\nTarget Transform\n\t{target_transform}"
        )

        self.transform = Compose(transform)
        self.target_transform = Compose(target_transform)

        # - Class balancing :  Weighted Random Sampler
        self.auto_balance_train_set = auto_balance_train_set
        self.auto_balance_val_set = auto_balance_val_set
        self.auto_balance_test_set = auto_balance_test_set

        logging.info(
            "Automatic Class Balancing via Weighted Random Sampler applied:"
            + f"\nTrain set\t{auto_balance_train_set}"
            + f"\nVal set\t{auto_balance_val_set}"
            + f"\nTest set\t{auto_balance_test_set}"
        )


    def setup(self, stage: Optional[str] = None) -> None:
        """Called at the beginning of fit (train + validate), validate, test, or predict. Lightning
        uses it.

        Args:
            stage (Optional[str], optional): either ``'fit'``, ``'validate'``, ``'test'``, or ``'predict'``. Defaults to None
        """
        self.train_data, self.train_sample_weights = self.__get_dataset(
            "train", self.auto_balance_train_set
        )
        self.val_data, self.val_sample_weights = self.__get_dataset(
            "val", self.auto_balance_val_set
        )
        self.test_data, self.test_sample_weights = self.__get_dataset(
            "test", self.auto_balance_test_set
        )

    def __get_dataset(
        self, partition: str, auto_balance: bool
    ) -> Tuple[Union[SpokenDigits, DiskCachedDataset], Optional[np.ndarray]]:
        """Prepare the dataset and make it ready for caching.

        Args:
            partition (str): either ``'train'``, ``'val'``, or ``'test'``

        Returns:
            Tuple[Union[SpokenDigits, DiskCachedDataset], Optional[np.ndarray]]:
                dataset: a wrappper around ``SpokenDigits`` or the ``SpokenDigits`` itself. Save the processed samples to hard-drive for the sake of re-usability.
                sample_weights: sample weights for `WeightedRandomSampler`
        """
        if partition not in ["train", "val", "test"]:
            raise ValueError(
                "Unrecognized partition: 'train', 'val' and 'test' available!"
            )

        dataset = SpokenDigits(
            transform=self.transform,
            target_transform=self.target_transform,
            partition=partition,
            **self.params_dataset,
        )

        sample_weights = (
            self.__create_sample_weights(dataset, partition) if auto_balance else None
        )

        if self.direct_cached_data:
            cache_dir = os.path.join(self.direct_cached_data, partition)
            dataset = DiskCachedDataset(
                                        dataset=dataset,
                                        cache_path=cache_dir,
                                        reset_cache=self.reset_cache,
                                    )
        elif self.cache_dataset:
            # - Cache ID: create cache id based on the parameters used in the dataset
            not_cacheing_params = ['partition','config_str','config_file','filenames','labels','nr_files', 'split_config_file']
            self.cache_id = create_config_str(params_dict = dataset.__dict__, not_cacheing_params = not_cacheing_params,cache_path = self.cache_path)

            dataset = DiskCachedDataset(
                dataset=dataset,
                cache_path=self.__get_cache_dir(partition),
                reset_cache=self.reset_cache,
            )
        return dataset, sample_weights

    def __create_sample_weights(
        self, dataset: SpokenDigits, partition: str
    ) -> np.ndarray:
        """Process the dataset and obtain the sample weights.

        Args:
            dataset (SpokenDigits): Target dataset
            partition (str): one of ["train", "val", "test"]

        Returns:
            np.ndarray: Sample weights for `WeightedRandomSampler`
        """
        sound_labels = np.array(dataset.labels)
        unique_sound_labels, num_samples_sound = np.unique(sound_labels, return_counts=True)

        # Set probabilities of the samples (do not need to add up to 1.0)
        sample_weights_sound = np.zeros_like(sound_labels, dtype=float)
        batch_call_probability = np.zeros_like(sound_labels, dtype=float)
        for label in unique_sound_labels:
            sample_weights_sound[sound_labels == label] = 1.0 / num_samples_sound[label]

            label_location = np.where(label == self.batch_class_distribution[:,0])[0][0]
            batch_call_probability[sound_labels == label] = self.batch_class_distribution[label_location,1]

        sample_weights = batch_call_probability * sample_weights_sound

        return sample_weights

    def __get_cache_dir(self, partition: str) -> str:
        """Create a cache directory.

        Args:
            partition (str): subfolder name for a partition.

        Returns:
            str: a cache directory like ``'/home/ugurcan/Documents/Baby-crying-dataset/cache/865fa4eea41e53c9c9a22a8b9b74c49c/train'``
        """
        __dir = self.cache_path
        cache_dir = os.path.join(__dir, "cache", self.cache_id, partition)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        return cache_dir

    def __get_dataloader(
        self,
        dataset: Union[SpokenDigits, DiskCachedDataset],
        sample_weights: Optional[np.ndarray],
        shuffle: bool,
        batch_size
    ) -> DataLoader:
        """Construct a dataloader with an optional ``WeightedRandomSampler``

        Args:
            dataset (Union[SpokenDigits, DiskCachedDataset]): the source dataset
            sample_weights (Optional[np.ndarray]): ``WeightedRandomSampler`` weights
            shuffle (bool): shuffle the sample order or not

        Returns:
            DataLoader: the dataloader for the partition "train", "val" or "test"
        """
        if sample_weights is not None:
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(dataset))
            shuffle = False
        else:
            sampler = None

        return DataLoader(
            dataset=dataset,
            num_workers=self.num_workers,
            batch_size=batch_size,
            prefetch_factor=self.prefetch_factor,
            shuffle=shuffle,
            sampler=sampler,
        )

    def train_dataloader(self) -> DataLoader:
        return self.__get_dataloader(
            self.train_data, self.train_sample_weights, shuffle=True, batch_size = self.batch_size_train
        )

    def val_dataloader(self) -> DataLoader:
        return self.__get_dataloader(
            self.val_data, self.val_sample_weights, shuffle=False, batch_size=self.batch_size_val
        )

    def test_dataloader(self) -> DataLoader:
        return self.__get_dataloader(
            self.test_data, self.test_sample_weights, shuffle=False, batch_size=self.batch_size_test
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import os
    from training_utils.cache_utils import create_config_str
    from tonic.audio_transforms import FixLength
    from training_utils.data_transforms import SwapAxes, AFESim3
    from data.dataset import DATA_PATH

    # - Specify cache data folder
    cache_path = os.path.join(DATA_PATH, "cache_data")

    # - Specify laguage of the spoken digits (english or german)
    language = "english"

    # - Specify digits to be used.
    # - Since we want to train an SNN that can be mapped onto Xylo, we will only choose the first 8 spoken digits
    digits=[0,1,2,3,4,5,6,7,8,9,10]

    # - Specify the percentage of train-val-test
    # the sum of the percentage for train, val, and test must be equal to 1
    ps = [0.8, 0.1, 0.1]

    # - Specify the time in seconds for each file in each partition
    split_time_per_partition = [0.5,0.5,0.5] #(0.5 seconds for each partition: train, val and test)

    #- Specify the sampling frerquency of the audios
    sampling_freq = 48000

    params_dataset ={"digits":digits,
                     "language": language,
                     "sample_rate":sampling_freq,
                     "percentage_train":ps[0],
                     "percentage_val":ps[1],
                     "percentage_test":ps[2],
                     "split_time_per_partition":split_time_per_partition}


    transform = [
        FixLength(length=int(sampling_freq * split_time_per_partition[0]), axis=1),
        AFESim3(fs=sampling_freq, dt=0.009994, spike_gen_mode="divisive_norm"),
        SwapAxes(),
    ]


    dl = SpokenDigitsDataModule(
        batch_size_train = 4500, batch_size_val = 500, batch_size_test = 500,
        params_dataset = params_dataset,
        transform=transform,
        cache_dataset=True, 
        cache_path = cache_path,
        reset_cache=False, 
    )
    dl.setup()

    train_dataloader = dl.train_dataloader()
    data_train, label_train = next(iter(train_dataloader))

    print("data shape:", data_train.shape)
    print()
    print("labels", label_train)
    print(len(label_train))

   
    val_dl = dl.val_dataloader()
    data_val, label_val = next(iter(val_dl))

    print("val data shape:", data_val.shape)
    print("val label shape:", len(label_val))
