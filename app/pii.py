import re
from app.deps import nlp_ru, is_valid_snils, is_valid_iin, region_map, analyzer, tox_model, tox_tokenizer
import torch

def extract_pii(text: str, lang: str):
    results = []
    
    text_lower = text.lower()

    if lang == 'ru':
        for sent in nlp_ru(text_lower).sentences:
            for ent in sent.ents:
                span = text_lower[ent.start_char:ent.end_char]
                if ent.type in ("LOC", "GPE") and span.upper().startswith(("СНИЛС", "ПАСПОРТ")):
                    continue
                if ent.type == "PER":
                    et = "PERSON"
                elif ent.type == "ORG":
                    et = "ORG"
                elif ent.type in ("LOC", "GPE"):
                    et = "GPE"
                else:
                    continue
                results.append({
                    "type": et,
                    "text": span
                })

        snils_patterns = [
            re.compile(r"\b\d{3}-\d{3}-\d{3}\s?\d{2}\b"),
            re.compile(r"\b\d{11}\b"),
            re.compile(r"\b\d{9}\s\d{2}\b")
        ]
        for rx in snils_patterns:
            for m in rx.finditer(text):
                span = text[m.start():m.end()]
                if is_valid_snils(span):
                    results.append({
                        "type": "RUS_SNILS",
                        "text": span
                    })

        iin_pattern = re.compile(r"\b\d{12}\b")
        for m in iin_pattern.finditer(text):
            span = m.group()
            if is_valid_iin(span):
                results.append({
                    "type": "KZ_IIN",
                    "text": span
                })

        phone_rx = re.compile(r"(?:\+7|8)\d{10}")
        for m in phone_rx.finditer(text):
            raw = m.group()
            norm = "+7" + raw[1:] if raw.startswith("8") else raw
            code = norm[2:5]

           
            kz_mobile_prefixes = ["700", "701", "702", "705", "707", "708", "747", "771", "775", "776", "777", "778"]
            
            if code in kz_mobile_prefixes:
                region = "Kazakhstan Mobile"
            elif code in region_map:
                region = region_map[code]
            else:
                continue
            
            results.append({
                "type": f"PHONE (region: {region})",
                "text": norm
            })

        email_results = analyzer.analyze(text=text, entities=["EMAIL_ADDRESS"], language="en")
        for e in email_results:
            results.append({
                "type": "EMAIL_ADDRESS",
                "text": text[e.start:e.end]
            })
    else:
        pii = analyzer.analyze(text=text, language='en')
        for e in pii:
            results.append({
                "type": e.entity_type,
                "text": text[e.start:e.end]
            })
    return results

def moderate_text(text: str, lang: str):
    has_pii = len(extract_pii(text, lang)) > 0

    inputs = tox_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = tox_model(**inputs)
        probs = outputs.logits.softmax(dim=1)[0]
        toxic_score = float(probs[1])
    is_toxic = toxic_score >= 0.1 
    return has_pii, is_toxic