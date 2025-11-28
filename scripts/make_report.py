from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_MD = ROOT / 'REPORT.md'
PLOTS = ROOT / 'outputs' / 'plots'
TEXT_PROC = ROOT / 'data' / 'processed' / 'text'
FER_PROC = ROOT / 'data' / 'processed' / 'fer2013'


def section(title: str) -> str:
    return f"\n### {title}\n\n"


def bullets(items):
    return ''.join([f"- {it}\n" for it in items])


def summarize_text() -> str:
    md = section('Text Emotion Dataset (provided CSV)')
    cleaned = TEXT_PROC / 'cleaned.csv'
    if not cleaned.exists():
        md += 'Cleaned dataset not found. Run `python scripts/eda_and_prepare_text.py` first.\n'
        return md
    df = pd.read_csv(cleaned)
    md += bullets([
        f"Total samples (cleaned): {len(df):,}",
        f"Classes: {', '.join(sorted(df['emotion'].unique()))}",
    ])
    # Class distribution
    vc = df['emotion'].value_counts()
    md += '\nClass distribution (top 10):\n\n'
    md += vc.to_string() + "\n\n"
    # Plots
    if (PLOTS / 'text_emotion_distribution.png').exists():
        md += f"![](outputs/plots/text_emotion_distribution.png)\n\n"
    if (PLOTS / 'text_length_hist.png').exists():
        md += f"![](outputs/plots/text_length_hist.png)\n\n"
    if (PLOTS / 'text_wordcloud.png').exists():
        md += f"![](outputs/plots/text_wordcloud.png)\n\n"
    return md


def summarize_fer() -> str:
    md = section('Face Emotion Dataset (FER2013)')
    train = FER_PROC / 'train.csv'
    val = FER_PROC / 'val.csv'
    test = FER_PROC / 'test.csv'
    if train.exists() and test.exists():
        df_train = pd.read_csv(train)
        df_val = pd.read_csv(val) if val.exists() else pd.DataFrame(columns=['path', 'label'])
        df_test = pd.read_csv(test)
        md += bullets([
            f"Train: {len(df_train):,} images",
            f"Val: {len(df_val):,} images",
            f"Test: {len(df_test):,} images",
        ])
        # Class dist plot
        if (PLOTS / 'fer_class_distribution.png').exists():
            md += f"\n![](outputs/plots/fer_class_distribution.png)\n\n"
        elif (PLOTS / 'fer_class_distribution_csv.png').exists():
            md += f"\n![](outputs/plots/fer_class_distribution_csv.png)\n\n"
        # Sample grids
        for name in ['fer_sample_grid.png', 'fer_sample_grid_csv.png', 'fer_channel_counts.png', 'fer_img_width_hist.png', 'fer_img_height_hist.png']:
            if (PLOTS / name).exists():
                md += f"![](outputs/plots/{name})\n\n"
    else:
        md += 'Prepared splits not found. Run: `python scripts/download_fer2013.py` then `python scripts/eda_and_prepare_fer.py`.\n'
    return md


def main():
    md = """### Face Mood Recognition — Data Report\n\nThis report summarizes data collection, cleaning, exploratory data analysis (EDA), and preparation for two datasets:\n- Face emotions: FER2013 (images)\n- Text emotions: Provided CSV (`Emotion_classify_Data.csv`)\n\n"""
    md += summarize_fer()
    md += summarize_text()

    OUT_MD.write_text(md, encoding='utf-8')
    print(f"Wrote report to {OUT_MD}")


if __name__ == '__main__':
    main()
