source .venv/bin/activate
python src/dataset_prepare.py

python src/emotions.py --mode train
python src/emotions.py --mode display
