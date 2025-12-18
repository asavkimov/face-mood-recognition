import numpy as np
import pandas as pd 
from PIL import Image
from tqdm import tqdm
import os
import sys
from pathlib import Path

# convert string to integer
def atoi(s):
    n = 0
    for i in s:
        n = n*10 + ord(i) - ord("0")
    return n

BASE_DIR = Path(__file__).resolve().parent
csv_path = BASE_DIR / 'fer2013.csv'

# making folders under src/data/...
outer_names = ['test','train']
inner_names = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
data_root = BASE_DIR / 'data'
train_root = data_root / 'train'
test_root = data_root / 'test'
os.makedirs(train_root, exist_ok=True)
os.makedirs(test_root, exist_ok=True)
for outer_name in outer_names:
    for inner_name in inner_names:
        os.makedirs(data_root / outer_name / inner_name, exist_ok=True)

# to keep count of each category
angry = disgusted = fearful = happy = sad = surprised = neutral = 0
angry_test = disgusted_test = fearful_test = happy_test = sad_test = surprised_test = neutral_test = 0

if not csv_path.exists():
    sys.exit(f"fer2013.csv not found at {csv_path}")

df = pd.read_csv(csv_path)

required_cols = {'pixels', 'emotion'}
missing = required_cols - set(df.columns)
if missing:
    sys.exit(f"Expected columns {required_cols}, but missing {missing}. "
             "Please place the original FER-2013 CSV with 'emotion,pixels,Usage' columns.")

mat = np.zeros((48,48),dtype=np.uint8)
print("Saving images...")

# read the csv file line by line
for i in tqdm(range(len(df))):
    txt = df['pixels'][i]
    words = txt.split()
    
    # the image size is 48x48
    for j in range(2304):
        xind = j // 48
        yind = j % 48
        mat[xind][yind] = atoi(words[j])

    img = Image.fromarray(mat)

    # train
    if i < 28709:
        if df['emotion'][i] == 0:
            img.save(train_root / 'angry' / f'im{angry}.png')
            angry += 1
        elif df['emotion'][i] == 1:
            img.save(train_root / 'disgusted' / f'im{disgusted}.png')
            disgusted += 1
        elif df['emotion'][i] == 2:
            img.save(train_root / 'fearful' / f'im{fearful}.png')
            fearful += 1
        elif df['emotion'][i] == 3:
            img.save(train_root / 'happy' / f'im{happy}.png')
            happy += 1
        elif df['emotion'][i] == 4:
            img.save(train_root / 'sad' / f'im{sad}.png')
            sad += 1
        elif df['emotion'][i] == 5:
            img.save(train_root / 'surprised' / f'im{surprised}.png')
            surprised += 1
        elif df['emotion'][i] == 6:
            img.save(train_root / 'neutral' / f'im{neutral}.png')
            neutral += 1
 
    # test
    else:
        if df['emotion'][i] == 0:
            img.save(test_root / 'angry' / f'im{angry_test}.png')
            angry_test += 1
        elif df['emotion'][i] == 1:
            img.save(test_root / 'disgusted' / f'im{disgusted_test}.png')
            disgusted_test += 1
        elif df['emotion'][i] == 2:
            img.save(test_root / 'fearful' / f'im{fearful_test}.png')
            fearful_test += 1
        elif df['emotion'][i] == 3:
            img.save(test_root / 'happy' / f'im{happy_test}.png')
            happy_test += 1
        elif df['emotion'][i] == 4:
            img.save(test_root / 'sad' / f'im{sad_test}.png')
            sad_test += 1
        elif df['emotion'][i] == 5:
            img.save(test_root / 'surprised' / f'im{surprised_test}.png')
            surprised_test += 1
        elif df['emotion'][i] == 6:
            img.save(test_root / 'neutral' / f'im{neutral_test}.png')
            neutral_test += 1

print("Done!")