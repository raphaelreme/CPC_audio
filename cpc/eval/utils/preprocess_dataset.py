import argparse
import json
import os
import random

import pandas
import torchaudio
import phonemizer


DB_IN = "cv-corpus-6.1-2020-12-11"
DB_OUT = "common_voice_preprocessed"
SAMPLING_RATE = 16000
TRAIN_SMALL_DURATION = 600
TRAIN_MEDIUM_DURATION = 3600
TRAIN_LARGE_DURATION = 36000
VAL_DURATION = 3600
TEST_SMALL_DURATION = 3600


def split(lang, seed=42):  # Ugly and repetitive...
    print("Splitting")
    os.makedirs(f"{DB_OUT}/{lang}", exist_ok=False)
    os.makedirs(f"{DB_OUT}/{lang}/clips")

    train_set = set(pandas.read_csv(f"{DB_IN}/{lang}/train.tsv", sep="\t")["path"])
    val_set = set(pandas.read_csv(f"{DB_IN}/{lang}/dev.tsv", sep="\t")["path"])
    test_set = set(pandas.read_csv(f"{DB_IN}/{lang}/test.tsv", sep="\t")["path"])
    extra_set = set(pandas.read_csv(f"{DB_IN}/{lang}/validated.tsv", sep="\t")["path"])
    extra_set = extra_set.difference(train_set).difference(val_set).difference(test_set)

    train_set = sorted(train_set)
    val_set = sorted(val_set)
    test_set = sorted(test_set)
    extra_set = sorted(extra_set)

    random.seed(seed)
    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)
    random.shuffle(extra_set)

    train = []
    total_length = 0
    for path in train_set + extra_set:
        x, sampling_rate = torchaudio.load(f"{DB_IN}/{lang}/clips/{path}")

        # Resample
        transform = torchaudio.transforms.Resample(
            orig_freq=sampling_rate, new_freq=SAMPLING_RATE, resampling_method='sinc_interpolation')
        x = transform(x)
        torchaudio.save(f"{DB_OUT}/{lang}/clips/{path}", x, SAMPLING_RATE, channels_first=True)

        length = x.shape[1] / SAMPLING_RATE

        if total_length < TRAIN_SMALL_DURATION and total_length + length >= TRAIN_SMALL_DURATION:
            print("\rSmall train split duration:", total_length)
            save_split(lang, train, "train_small.txt")

        if total_length < TRAIN_MEDIUM_DURATION and total_length + length >= TRAIN_MEDIUM_DURATION:
            print("\rMedium train split duration:", total_length)
            save_split(lang, train, "train_medium.txt")

        if total_length < TRAIN_LARGE_DURATION and total_length + length >= TRAIN_LARGE_DURATION:
            print("\rLarge train split duration:", total_length)
            save_split(lang, train, "train_large.txt")
            break

        train.append(path[:-4])  # Add path without .mp3
        total_length += length

        print(f"\r{total_length:.0f}/{TRAIN_LARGE_DURATION} ....", end="", flush=True)

    val = []
    total_length = 0
    for path in val_set:
        x, sampling_rate = torchaudio.load(f"{DB_IN}/{lang}/clips/{path}")

        # Resample
        transform = torchaudio.transforms.Resample(
            orig_freq=sampling_rate, new_freq=SAMPLING_RATE, resampling_method='sinc_interpolation')
        x = transform(x)
        torchaudio.save(f"{DB_OUT}/{lang}/clips/{path}", x, SAMPLING_RATE, channels_first=True)

        length = x.shape[1] / sampling_rate

        if total_length < VAL_DURATION and total_length + length >= VAL_DURATION:
            print("\rVal split duration:", total_length)
            save_split(lang, val, "val.txt")
            break

        val.append(path[:-4])  # Add path without .mp3
        total_length += length

        print(f"\r{total_length:.0f}/{VAL_DURATION} ....", end="", flush=True)

    test = []
    total_length = 0
    for i, path in enumerate(test_set):
        x, sampling_rate = torchaudio.load(f"{DB_IN}/{lang}/clips/{path}")

        # Resample
        transform = torchaudio.transforms.Resample(
            orig_freq=sampling_rate, new_freq=SAMPLING_RATE, resampling_method='sinc_interpolation')
        x = transform(x)
        torchaudio.save(f"{DB_OUT}/{lang}/clips/{path}", x, SAMPLING_RATE, channels_first=True)

        length = x.shape[1] / SAMPLING_RATE

        if total_length < TEST_SMALL_DURATION and total_length + length >= TEST_SMALL_DURATION:
            print("\rSmall test split duration:", total_length)
            save_split(lang, test, "test_small.txt")

        test.append(path[:-4])  # Add path without .mp3
        total_length += length

        print(f"\r{i+1}/{len(test_set)} ....", end="", flush=True)

    print("\rFull test split duration:", total_length)
    save_split(lang, test, "test_full.txt")

    return train + val + test


def save_split(lang, files, name):
    with open(f"{DB_OUT}/{lang}/{name}", "w") as f:
        f.write("\n".join(files))



def phonemize(lang, files):
    print("Phonemize")
    validated_set = pandas.read_csv(f"{DB_IN}/{lang}/validated.tsv", sep="\t")
    paths = list(validated_set["path"])
    sentences = list(validated_set["sentence"])
    validated_dict = {paths[i][:-4]: sentences[i] for i in range(len(paths))}

    phonemes_to_id = {}
    annotations = []

    for i, file in enumerate(files):
        sentence = validated_dict[file]
        phonemes = phonemizer.phonemize(sentence, language=lang, backend="espeak")

        out = []
        for char in phonemes:
            if char == " ":
                continue
            try:
                out.append(str(phonemes_to_id[char]))
            except KeyError:
                phonemes_to_id[char] = len(phonemes_to_id)
                out.append(str(phonemes_to_id[char]))

        line = file + " " + " ".join(out)
        annotations.append(line)
        print(f"\r{i+1}/{len(files)} ....", end="", flush=True)
    print(f"\r                           \r", end="", flush=True)

    with open(f"{DB_OUT}/{lang}/annotations.txt", "w") as f:  # Annotation file
        f.write("\n".join(annotations))

    with open(f"{DB_OUT}/{lang}/phonemes_to_id.json", "w") as f:  # Phonemes to id
        json.dump(phonemes_to_id, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("lang")
    parser.add_argument("-i", "--db-in", default=DB_IN)
    parser.add_argument("-o", "--db_out", default=DB_OUT)
    parser.add_argument("-f", "--sampling-rate", type=int, default=SAMPLING_RATE)
    parser.add_argument("--train-small-duration", type=int, default=TRAIN_SMALL_DURATION)
    parser.add_argument("--train-medium-duration", type=int, default=TRAIN_MEDIUM_DURATION)
    parser.add_argument("--train-large-duration", type=int, default=TRAIN_LARGE_DURATION)
    parser.add_argument("--val-duration", type=int, default=VAL_DURATION)
    parser.add_argument("--test-small-duration", type=int, default=TEST_SMALL_DURATION)
    args = parser.parse_args()

    DB_IN = args.db_in
    DB_OUT = args.db_out
    SAMPLING_RATE = args.sampling_rate
    TRAIN_SMALL_DURATION = args.train_small_duration
    TRAIN_MEDIUM_DURATION = args.train_medium_duration
    TRAIN_LARGE_DURATION = args.train_large_duration
    VAL_DURATION = args.val_duration
    TEST_SMALL_DURATION = args.test_small_duration

    files = split(args.lang)
    phonemize(args.lang, files)
