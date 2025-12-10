import os
import json 
import glob
import random
import librosa
import logging
import hashlib
import numpy as np
import soundfile as sf
from dataclasses import dataclass, field
from typing import Tuple, Optional, Callable
from scipy.io.wavfile import write as wavwrite

DATA_PATH =  './path_to_the_data_folder/' # Add the path to your data here 

def default_digits():
    return [0,1,2,3,4,5,6,7,8,9]
def default_language():
    return "english"
def default_speakers():
    return ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11']

@dataclass
class SpokenDigits:
    partition: str = "train"
    digits: list = field(default_factory=default_digits)
    language: str = field(default_factory=default_language)
    speakers: list = field(default_factory=default_speakers)
    sample_rate: int = 48000
    percentage_train: float = 0.8
    percentage_val: float = 0.1
    percentage_test: float = 0.1
    transform: Optional[Callable] = None
    target_transform: Optional[Callable] = None
    split_time_per_partition: list = None

    """
        :param transform: transformation applied to signal data when item is loaded
        :param target_transform: transformation applied to target data when item is loaded
        :param sample_rate: the sampling frequency the data will be returned 
        :param partition: 'train', 'val', 'test'
        :param percentage_train: length of train set
        :param percentage_val: length of validation set
        :param percentage_test: lengths of test set
        :split_time_per_partition: the lenght in seconds that the data will be split for each partition
        :param digists: list of digit classes to use
        :param language: language of the spoken digits. Only "enlgish" and "german" are available
    """

    def __post_init__(self):
        # Set seed to garantee the same config file for multiple users of the dataset
        random.seed(123)
        # This is important to do, verify the sum of the percentage for train, val, and test is equal to 1
        if np.abs(self.percentage_train  + self.percentage_val + self.percentage_test - 1) > 1e-5:
            raise ValueError("percentages of train/val/test set need to add to one")
        # Create hash from all arguments and initialize train, test, val configuration files
        # This hash name allow us to know if we have previously generated the dataset with the 
        # given parameters and avoid rendundaces when calling the dataset. (More details in the description of each function)
        self.config_str = self.create_config_str()
        self.config_file = self.create_config_filename()       
        if self.split_time_per_partition == None:
            raise ValueError("a split time for each partition must be specified if load_split_data is True!")
        if len(self.split_time_per_partition) != 3:
            raise ValueError("a split time for each partition must be specified")
        self.split_config_file = self.create_split_config_filename()
        self.filenames, self.labels = self.load_data()
        self.nr_files = len(self.filenames)

    def create_config_str(self):
        """
        Creates a unique configuration string so we have control of the already separated data.
        All the fields in the dataclas are gathered by accessing 'self.__dataclass_fields__' and
        those fields that should not be part of the configuration string are removed.
        For example, when the configuration dictionary is created the first time, it will do it for both the train and 
        validation partition. When the user specifies the partition, however, this partition does not 
        make changes in the configuration dictionary and therefore, should be removed from the config string.

        Returns:
            config_str (str): string including all the parameters used to call the dataset
        """
        config_str = ""
        for field in self.__dataclass_fields__:
            # drop arguments that are not necessary to identify unique configuration
            if field not in [
                "transform",
                "target_transform",
                "partition",
                "split_time_per_partition",
            ]:
                config_str += field + str(getattr(self, field))

        return config_str

    def create_config_filename(self):
        # hashlib.sha1() is what will convert the config_str into bjf4h35j5b43b34, for example
        filename = f"{DATA_PATH}/config_{hashlib.sha1(self.config_str.encode('utf-8')).hexdigest()}.json"
        return filename

    def create_split_config_filename(self):
        config_str = ""
        for duration in self.split_time_per_partition:
            config_str += str(duration)
        filename = f"{DATA_PATH}/config_{hashlib.sha1(self.config_str.encode('utf-8')).hexdigest()}_{hashlib.sha1(config_str.encode('utf-8')).hexdigest()}.json"
        return filename

    def randomize_files(self, config_file):
        # Since we read the audios in a systematic way, we need to make sure we shuffle the data:
        for partition_key in config_file.keys(): 
            random_vector = np.arange(0,len(config_file[f"{partition_key}"][0]),1)
            random.shuffle(random_vector)
            # Shuffle the files in the partition
            config_file[f"{partition_key}"][0] = (np.asarray(config_file[f"{partition_key}"][0])[random_vector]).tolist()
            # With the same random vector, shuffle the label id
            config_file[f"{partition_key}"][1] = (np.asarray(config_file[f"{partition_key}"][1])[random_vector]).tolist()
        
        return config_file

    def load_data(self):
        try:
            with open(self.config_file) as f:
                logging.info(f"\nLoading config file according to the percentage: train = {self.percentage_train} val = {self.percentage_val} test ={ self.percentage_test}")
                config = json.load(f)
                logging.info(f"\nVerifying if data has been split in {self.split_time_per_partition}seconds for train-val-test...")
                try:
                    with open(self.split_config_file) as f:
                        split_config = json.load(f)
                        randomized_config_file = self.randomize_files(split_config)
                        files, labels = randomized_config_file[self.partition]
                    logging.info(
                    f"\nA data split of {self.split_time_per_partition}seconds for train-val-test has been generated before"
                    + "\nLoading this split partition..."
                    + "\nSplit data is ready"
                    )
                except:
                    logging.info(f"\nThe data split of {self.split_time_per_partition}seconds for train-val-test has not been generated before")
                    self.split_data(config)
                    with open(self.split_config_file) as f:
                        split_config = json.load(f)
                        randomized_config_file = self.randomize_files(split_config)
                        files, labels = randomized_config_file[self.partition]
                    logging.info("Split data is ready")

        except:
            logging.info(f"\nGenerating parition train:{self.percentage_train} val: {self.percentage_val} test:{ self.percentage_test}")
            self.partition_data()
            files, labels = self.load_data()

        return files, labels

    def partition_data(self):
        # dictionary with list of filenames and labels for each partition
        config = {"train": [], "val": [], "test": []}

        # Split the subjects into train-val-test
        random.shuffle(self.speakers)
        num_subjects = len(self.speakers)
        n_test_subjects = round(self.percentage_test*num_subjects)
        n_val_subjects = round(self.percentage_val*num_subjects)

        test_subjects = self.speakers[:n_test_subjects]
        val_subjects = self.speakers[n_test_subjects:n_test_subjects+n_val_subjects]
        train_subjects = self.speakers[(n_test_subjects+n_test_subjects):]

        subjects_per_partition = [train_subjects,val_subjects,test_subjects]
        
        # for each class partition data
        for partition, subjects in zip(list(config.keys()),subjects_per_partition):  
            logging.info(
                "\n---------------------------------------------------------------------------"
                + f"\nCreating {partition} partition"
                )
            for subj in subjects:
                logging.info(f"\nAccessing files from class speaker {subj}")
                for digit in self.digits:
                    files_speaker = glob.glob(f"{DATA_PATH}/audio/*{self.language}*{subj}*digit-{digit}.flac", recursive=True)  
                    random.shuffle(files_speaker)
                    config[partition] += files_speaker
            
        # We save the config file. This file will only be used to know which speaker are in each partition. 
        # However, it can not be used for training since no label was saved and all the files are of different duration.
        with open(self.config_file, "w") as f:
            json.dump(config, f)

        return config[self.partition]

    def split_data(self, partition_config_file):
        # Create config file for the split data
        split_config_file = {"train": [[], []], "val": [[], []], "test": [[], []]}

        # Create folder to save split audios
        split_data_path = os.path.join(DATA_PATH, 'split_audios')
        if not os.path.exists(split_data_path):
            os.mkdir(split_data_path)
        split_folder = self.split_config_file.split('/')[-1].split('.json')[0]
        split_data_path = os.path.join(split_data_path,split_folder)
        if not os.path.exists(split_data_path):
            os.mkdir(split_data_path)
            
        # Check if split partition exist
        logging.info("Creating split from the config file")

        # Split the data per partition    
        for p, partition_key in enumerate(partition_config_file.keys()):
            discarded_sample =np.zeros(len(self.digits))
            # Create partition folder
            partition_split_data_path = os.path.join(split_data_path,partition_key)
            if not os.path.exists(partition_split_data_path):
                os.mkdir(partition_split_data_path)

            samples_split_time = int(self.split_time_per_partition[p] * self.sample_rate)
            logging.info(f"Splitting data from partition: {partition_key}")
            # For each file in the config file
            for file in partition_config_file[partition_key]:
                # Get all_the info from the sample to save it in the right location
                speaker_id = file.split('speaker-')[-1].split('_')[0]
                trial = file.split('trial-')[-1].split('_')[0]
                digit = file.split('digit-')[-1].split('.flac')[0]
                # Read the data
                data, _ = sf.read(file) 
                num_splits = int(np.floor(data.size/samples_split_time))
                if num_splits <=0:
                    discarded_sample[int(digit)]+=1
                start_time = 0
                end_time = samples_split_time
                for s in range(num_splits):
                    split_data = data[start_time:end_time]
                    start_time+=samples_split_time
                    end_time+=samples_split_time
                    # Save the split data
                    new_sample_name = file.split('/')[-1].split('.flac')[0] + f'_split-{s}.wav'
                    wavwrite(filename=os.path.join(partition_split_data_path,new_sample_name),
                             rate=self.sample_rate,
                             data=split_data.astype(np.float32),
                            )
                    # Append the name in the split config file together with the noise label
                    split_config_file[f"{partition_key}"][0] += [os.path.join(partition_split_data_path,new_sample_name)]
                    split_config_file[f"{partition_key}"][1] += [int(digit)]

            for digit, digit_label in enumerate(self.digits):
                logging.info(
                f"\nFor class {digit_label} in {partition_key} partition:"
                + f"\nA total of {discarded_sample[digit]} events were rejected for being shorter than the split time of {self.split_time_per_partition[p]} seconds"
                )

        with open(self.split_config_file, "w") as f:
            json.dump(split_config_file, f)

    def __getitem__(self, item) -> Tuple[Tuple[np.ndarray, int], int]:
        # This functions should reamin as it is
        # datamodule will cann this fucntion to get the data, so it should
        # return the transformed signal and label.
        label = int(self.labels[item])
        signal, sample_rate = librosa.core.load(
            self.filenames[item], sr=self.sample_rate
        )

        if signal.ndim == 1:
            # Introduce a channel dimension
            signal = signal[None, ...]  
        if self.transform is not None:
            signal = self.transform(signal)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return signal, label

    def __len__(self) -> int:
        return self.nr_files


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ps = [0.8, 0.1, 0.1]
    data_train = SpokenDigits(
        partition="train",
        percentage_train=ps[0],
        percentage_val=ps[1],
        percentage_test=ps[2],
        split_time_per_partition = [0.5,0.5,0.5],
    )

    print(data_train.__len__())
    test = data_train.__getitem__(0)
    signal = test[0]
