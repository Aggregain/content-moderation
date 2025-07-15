from pathlib import Path
import pandas as pd
import stanza
from presidio_analyzer import AnalyzerEngine
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from kz_iin_validator import validate_iin

def is_valid_snils(snils: str) -> bool:
    digits = ''.join(ch for ch in snils if ch.isdigit())
    if len(digits) != 11:
        return False
    nums = list(map(int, digits))
    checksum_value = nums[-2] * 10 + nums[-1]
    total = sum((i + 1) * nums[i] for i in range(9))
    if total < 100:
        control = total
    elif total in (100, 101):
        control = 0
    else:
        control = total % 101
        if control == 100:
            control = 0
    return control == checksum_value

def is_valid_iin(iin: str) -> bool:
    try:
        validate_iin(iin)
        return True
    except ValueError:
        return False

DATA_DIR = Path(__file__).parent.parent / "data"

PATTERNS_FILE = DATA_DIR / "russian_pii_patterns.xlsx"
REGIONS_FILE = DATA_DIR / "rus_phone_regions.xlsx"

regions_df = pd.read_excel(REGIONS_FILE, dtype=str)
regions_df["region_code"] = regions_df["region_code"].str.strip()
region_map = dict(zip(regions_df["region_code"], regions_df["region_name"]))

stanza.download("ru", verbose=False)
nlp_ru = stanza.Pipeline(lang="ru", processors="tokenize,ner", verbose=False)

analyzer = AnalyzerEngine()

tox_model_name = "textdetox/xlmr-large-toxicity-classifier-v2"
tox_model = AutoModelForSequenceClassification.from_pretrained(tox_model_name)
tox_tokenizer = AutoTokenizer.from_pretrained(tox_model_name)