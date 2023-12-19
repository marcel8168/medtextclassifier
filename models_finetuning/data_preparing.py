import pandas as pd
from dataset import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from global_parameters import RANDOM_SEED


def get_dataloader(texts, targets, tokenizer, batch_size, max_len, num_workers=0):
    dataset = Dataset(texts.to_numpy(), targets, tokenizer, max_len)
    params = {
        "batch_size": batch_size,
        "num_workers": num_workers
    }
    dataloader = DataLoader(dataset, **params)

    return dataloader


def split_data(hum_df, vet_df, max_size):
    hum_df = hum_df.sample(max_size, random_state=42).reset_index(
        drop=True, inplace=False)
    vet_df = vet_df.sample(max_size, random_state=42).reset_index(
        drop=True, inplace=False)

    hum_train_set, hum_test_set = train_test_split(
        hum_df,
        test_size=0.1,
        random_state=RANDOM_SEED
    )

    hum_train_set, hum_val_set = train_test_split(
        hum_train_set,
        test_size=0.2,
        random_state=RANDOM_SEED
    )

    vet_train_set, vet_test_set = train_test_split(
        vet_df,
        test_size=0.1,
        random_state=RANDOM_SEED
    )

    vet_train_set, vet_val_set = train_test_split(
        vet_train_set,
        test_size=0.2,
        random_state=RANDOM_SEED
    )

    train_set = pd.concat([hum_train_set, vet_train_set]).sample(
        frac=1).reset_index(drop=True, inplace=False)
    val_set = pd.concat([hum_val_set, vet_val_set]).sample(
        frac=1).reset_index(drop=True, inplace=False)
    test_set = pd.concat([hum_test_set, vet_test_set]).sample(
        frac=1).reset_index(drop=True, inplace=False)

    return train_set, val_set, test_set
