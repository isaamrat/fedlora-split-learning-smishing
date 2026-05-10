# Data Directory — FedSmishGuard

## What Is Included

| Path | Size | Description |
|---|---|---|
| `splits/train.csv` | ~8MB | Training set (21,591 rows, stratified) |
| `splits/val.csv` | ~1.1MB | Validation set (3,085 rows) |
| `splits/test.csv` | ~2.3MB | Test set — contains near-duplicates (6,170 rows) |
| `splits/test_clean.csv` | ~1.8MB | Clean test — near-duplicates removed (5,192 rows) |
| `clients/setting_D_300/` | ~8MB | 5 client CSVs with smishing floor of 300 (best setting) |
| `processed/source_label_distribution.csv` | tiny | Per-source label counts |
| `processed/split_leakage_report.csv` | tiny | Leakage summary statistics |
| `processed/client_summary_all_settings.csv` | tiny | Client sizes across settings |
| `sample/sample_sms_dataset.csv` | tiny | 30-row sample for quick testing |

## What Is NOT Included (and Why)

| File | Reason | How to Recreate |
|---|---|---|
| `processed/unified_sms_dataset.csv` | 11.5MB — large for GitHub | Run `build_dataset.py` |
| `processed/duplicate_report.csv` | 1.2MB — intermediate output | Run `build_dataset.py` |
| `raw/` | Raw downloads are 100MB+ | Download from sources below |
| `clients/setting_B/`, `setting_A/` etc. | Multiple 8MB folders | Run `split_clients.py` |

## How to Recreate the Full Dataset

### 1. Download raw datasets

Place files in `data/raw/`:

| Dataset | URL | Expected filename |
|---|---|---|
| Mendeley Balanced Spam-Smishing 10191 | https://data.mendeley.com/datasets/f45bkkt8pr | `balanced_spam_smishing_10191.csv` |
| UCI SMS Spam Collection | https://archive.ics.uci.edu/dataset/228 | `SMSSpamCollection` |
| Combined Labeled SMS | Kaggle: sms-spam-collection-dataset | `combined_labeled_dataset.csv` |
| Super SMS Dataset | GitHub/Kaggle (smishing-dataset) | `super_sms_dataset.csv` |
| ExAIS SMS Spam | GitHub: exais-sms | `exais_sms_spam.csv` |
| Mendeley 5971 | https://data.mendeley.com/datasets/tby3g5mb6n | `mendeley_5971.csv` |

### 2. Build the unified dataset

```bash
python src/build_dataset.py
# Output: data/processed/unified_sms_dataset.csv
```

### 3. Create splits

```bash
python src/build_dataset.py --split
# Output: data/splits/train.csv, val.csv, test.csv
```

### 4. Create client splits

```bash
# Setting D_300 (best — what models were trained on)
python src/split_clients.py --setting D --min_smishing 300 --out_dir data/clients/setting_D_300

# Setting B (standard Non-IID, no floor)
python src/split_clients.py --setting B --out_dir data/clients/setting_B
```

### 5. Create clean test (remove near-duplicates)

```bash
python src/near_duplicate_check.py   # generates reports/near_duplicate_pairs.csv
python src/create_clean_test.py      # generates data/splits/test_clean.csv
```

## Data Format

All CSV files use these columns:

| Column | Type | Description |
|---|---|---|
| `text` | str | Raw SMS message |
| `cleaned_text` | str | Preprocessed text (lowercased, URL/phone normalized) |
| `label` | str | `ham` / `spam` / `smishing` |
| `source` | str | Source dataset name |
| `category` | str | Smishing category (reward_prize, delivery_package, etc.) or NaN |

## Notes

- All splits are stratified by label
- Ham and spam are fully partitioned across clients (zero overlap between clients) in Setting B+
- Near-duplicate detection used MinHash LSH with Jaccard threshold 0.8, k=5 character shingles, 128 hash functions
- **Always evaluate on `test_clean.csv`** — the regular `test.csv` contains near-duplicates that inflate FNR by 8–10pp
