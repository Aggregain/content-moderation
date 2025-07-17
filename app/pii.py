import re
from app.deps import nlp_ru, is_valid_snils, is_valid_iin, region_map, analyzer, tox_model, tox_tokenizer
import torch

def extract_pii(text: str, lang: str):
    results = []
    found_spans = set()

    if lang == 'ru':
        doc = nlp_ru(text)
        for sent in doc.sentences:
            for ent in sent.ents:
                span = text[ent.start_char:ent.end_char]

                if ent.type in ("LOC", "GPE") and span.upper().startswith(("СНИЛС", "ПАСПОРТ")):
                    continue

                entity_type = None
                if ent.type == "PER":
                    entity_type = "PERSON"
                elif ent.type == "ORG":
                    entity_type = "ORG"
                elif ent.type in ("LOC", "GPE"):
                    entity_type = "GPE"

                if entity_type and span.lower() not in found_spans:
                    results.append({"type": entity_type, "text": span})
                    found_spans.add(span.lower())

        name_patterns = [
            re.compile(r"(?:меня зовут|моё имя|мое имя|зовут меня)\s+([А-Яа-яёЁ]+)", re.IGNORECASE)
        ]
        for pattern in name_patterns:
            for match in pattern.finditer(text):
                name = match.group(1)
                if name.lower() not in found_spans:
                    results.append({
                        "type": "PERSON",
                        "text": name
                    })
                    found_spans.add(name.lower())

        snils_patterns = [
            re.compile(r"\b\d{3}-\d{3}-\d{3}\s?\d{2}\b"),
            re.compile(r"\b\d{11}\b"),
            re.compile(r"\b\d{9}\s\d{2}\b")
        ]
        for rx in snils_patterns:
            for m in rx.finditer(text):
                span = text[m.start():m.end()]
                if is_valid_snils(span):
                    if span.lower() not in found_spans:
                        results.append({"type": "RUS_SNILS", "text": span})
                        found_spans.add(span.lower())

        iin_pattern = re.compile(r"\b\d{12}\b")
        for m in iin_pattern.finditer(text):
            span = m.group()
            if is_valid_iin(span):
                if span.lower() not in found_spans:
                    results.append({"type": "KZ_IIN", "text": span})
                    found_spans.add(span.lower())

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

            if norm.lower() not in found_spans:
                results.append({"type": f"PHONE (region: {region})", "text": norm})
                found_spans.add(norm.lower())

        email_results = analyzer.analyze(text=text, entities=["EMAIL_ADDRESS"], language="en")
        for e in email_results:
            span = text[e.start:e.end]
            if span.lower() not in found_spans:
                results.append({"type": "EMAIL_ADDRESS", "text": span})
                found_spans.add(span.lower())
    else:
        pii = analyzer.analyze(text=text, language='en')
        for e in pii:
            results.append({
                "type": e.entity_type,
                "text": text[e.start:e.end]
            })
    return results

def normalize_for_toxicity(text: str, lang: str):
    if lang != 'ru':
        return text

    doc = nlp_ru(text)
    if not doc.sentences:
        return text

    spans_to_lower = []
    for sent in doc.sentences:
        for ent in sent.ents:
            if ent.type in ("ORG", "GPE"):
                spans_to_lower.append((ent.start_char, ent.end_char))

    normalized = ""
    last = 0
    for start, end in sorted(spans_to_lower):
        normalized += text[last:start] + text[start:end].lower()
        last = end
    normalized += text[last:]
    return normalized



def moderate_text(text: str, lang: str):
    has_pii = len(extract_pii(text, lang)) > 0

    normalized_text = normalize_for_toxicity(text, lang)

    inputs = tox_tokenizer(normalized_text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = tox_model(**inputs)
        probs = outputs.logits.softmax(dim=1)[0]
        toxic_score = float(probs[1])
    is_toxic = toxic_score >= 0.1

    print(f"[DEBUG] Toxic score: {toxic_score:.4f}")
    return has_pii, is_toxic
