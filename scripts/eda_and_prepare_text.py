import re
import sys
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from wordcloud import WordCloud, STOPWORDS

PLOTS_DIR = Path('outputs/plots')
RAW_TEXT = Path('data/raw/text/Emotion_classify_Data.csv')
PROC_DIR = Path('data/processed/text')

EMOTION_ORDER = [
    'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'
]

# Map various label variants/synonyms to the target set
EMO_MAP = {
    'anger': 'angry', 'angry': 'angry',
    'disgust': 'disgust', 'disgusted': 'disgust',
    'fear': 'fear', 'scared': 'fear', 'terrified': 'fear', 'afraid': 'fear',
    'joy': 'happy', 'happiness': 'happy', 'joyful': 'happy', 'glad': 'happy',
    'sad': 'sad', 'sadness': 'sad', 'depressed': 'sad', 'unhappy': 'sad',
    'surprise': 'surprise', 'surprised': 'surprise', 'astonished': 'surprise',
    'neutral': 'neutral', 'none': 'neutral', 'no emotion': 'neutral', 'others': 'neutral'
}


def ensure_dirs():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ''
    s = s.lower()
    # Remove URLs
    s = re.sub(r'https?://\S+|www\.\S+', ' ', s)
    # Remove mentions/hashtags handles but keep the word
    s = re.sub(r'[@#](\w+)', r'\1', s)
    # Remove html entities
    s = re.sub(r'&\w+;', ' ', s)
    # Remove punctuation (keep apostrophes within words)
    s = re.sub(r"[^a-z0-9'\s]", ' ', s)
    # Collapse repeated whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def load_and_clean(raw_path: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    # Expect columns: comment, emotion
    col_comment = 'comment' if 'comment' in df.columns else df.columns[0]
    col_emotion = 'emotion' if 'emotion' in df.columns else df.columns[1]
    df = df[[col_comment, col_emotion]].rename(columns={col_comment: 'text', col_emotion: 'emotion'})
    # Clean
    df['text'] = df['text'].astype(str).map(clean_text)
    df['emotion'] = df['emotion'].astype(str).str.lower().str.strip()
    # Map label variants to the target set where possible
    df['emotion'] = df['emotion'].map(lambda x: EMO_MAP.get(x, x))
    # Drop empties
    df = df[(df['text'].str.len() > 0) & (df['emotion'].str.len() > 0)]
    # Filter to known emotions if present (after mapping)
    if set(df['emotion'].unique()) & set(EMOTION_ORDER):
        df = df[df['emotion'].isin(EMOTION_ORDER)]
    # Drop duplicates
    df = df.drop_duplicates(subset=['text', 'emotion']).reset_index(drop=True)
    return df


def plot_class_distribution(df: pd.DataFrame, title: str, out_path: Path):
    plt.figure(figsize=(8, 4))
    order = [e for e in EMOTION_ORDER if e in df['emotion'].unique()]
    sns.countplot(data=df, x='emotion', order=order, palette='Set2')
    plt.title(title)
    plt.ylabel('count')
    plt.xlabel('emotion')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_text_length_hist(df: pd.DataFrame, out_path: Path):
    lens = df['text'].str.split().map(len)
    plt.figure(figsize=(8, 4))
    sns.histplot(lens, bins=40, kde=False, color='#5DADE2')
    plt.title('Text length (tokens)')
    plt.xlabel('tokens per sample')
    plt.ylabel('count')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def top_tokens(df: pd.DataFrame, n=30) -> pd.DataFrame:
    def tokens(s):
        return re.findall(r"[a-z]+'?[a-z]*", s)
    counter = Counter()
    for t in df['text']:
        counter.update(tokens(t))
    items = counter.most_common(n)
    return pd.DataFrame(items, columns=['token', 'count'])


def plot_wordcloud(df: pd.DataFrame, out_path: Path):
    text = ' '.join(df['text'].tolist())
    wc = WordCloud(width=800, height=400, background_color='white', stopwords=STOPWORDS)
    img = wc.generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(img, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def stratified_splits(df: pd.DataFrame, seed: int = 42):
    # Create train (80%), temp (20%) then split temp into val/test (50/50)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, temp_idx = next(sss1.split(df, df['emotion']))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_idx, test_idx = next(sss2.split(temp_df, temp_df['emotion']))
    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)
    return train_df, val_df, test_df


def main():
    ensure_dirs()
    if not RAW_TEXT.exists():
        print(f"[text] Missing raw CSV at {RAW_TEXT}. Place the dataset there.")
        return 1

    df = load_and_clean(RAW_TEXT)
    # Save cleaned
    cleaned_csv = PROC_DIR / 'cleaned.csv'
    df.to_csv(cleaned_csv, index=False)

    # EDA plots
    plot_class_distribution(df, 'Emotion distribution (cleaned)', PLOTS_DIR / 'text_emotion_distribution.png')
    plot_text_length_hist(df, PLOTS_DIR / 'text_length_hist.png')
    try:
        plot_wordcloud(df, PLOTS_DIR / 'text_wordcloud.png')
    except Exception as e:
        print('[text] Wordcloud failed:', e)

    # Top tokens overall and per class
    top_overall = top_tokens(df, n=40)
    top_overall.to_csv(PROC_DIR / 'top_tokens_overall.csv', index=False)
    for emo in sorted(df['emotion'].unique()):
        tt = top_tokens(df[df['emotion'] == emo], n=30)
        tt.to_csv(PROC_DIR / f'top_tokens_{emo}.csv', index=False)

    # Splits
    train_df, val_df, test_df = stratified_splits(df)
    train_df.to_csv(PROC_DIR / 'train.csv', index=False)
    val_df.to_csv(PROC_DIR / 'val.csv', index=False)
    test_df.to_csv(PROC_DIR / 'test.csv', index=False)

    # Print brief report
    print('[text] Samples:', len(df))
    print('[text] Class distribution:')
    print(df['emotion'].value_counts().to_string())
    print('[text] Saved cleaned and splits under data/processed/text')
    print('[text] Plots saved to outputs/plots')
    return 0


if __name__ == '__main__':
    sys.exit(main())
