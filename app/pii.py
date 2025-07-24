import re
from app.deps import nlp_ru, is_valid_snils, region_map, analyzer, tox_model, tox_tokenizer
import torch

def extract_pii(text: str, lang: str):
    results = []
    
    if lang == 'ru':
        found_persons = []
        found_other_pii = []
        found_spans = set()

        doc = nlp_ru(text)

        entities = {}
        for sent in doc.sentences:
            for ent in sent.ents:
                span = text[ent.start_char:ent.end_char]
                if ent.type not in entities:
                    entities[ent.type] = []
                
                if ent.type in ("LOC", "GPE") and span.upper().startswith(("СНИЛС", "ПАСПОРТ")):
                    continue
                
                if span.lower() not in found_spans:
                    entities[ent.type].append(span)
                    found_spans.add(span.lower())
        
        if "PER" in entities:
            for item in entities["PER"]:
                found_persons.append({"type": "PERSON", "text": item})

  
        address_keywords = [
            'улица', 'ул.', 'проспект', 'пр-т', 'площадь', 'пл.', 'дом', 'д.', 
            'квартира', 'кв.', 'корпус', 'к.', 'строение', 'стр.', 'город', 'г.',
            'поселок', 'пос.', 'деревня', 'д.'
        ]
        address_pattern = re.compile(r'\b(' + '|'.join(address_keywords) + r')\b', re.IGNORECASE)
        if address_pattern.search(text):
            if "LOC" in entities or "GPE" in entities:
                full_address_parts = entities.get("LOC", []) + entities.get("GPE", [])
                full_address = ", ".join(part for part in full_address_parts if part not in found_persons)
                if full_address:
                    found_other_pii.append({"type": "ADDRESS", "text": full_address})
        
        if "GPE" in entities:
            for item in entities["GPE"]:
                found_other_pii.append({"type": "BIRTH_PLACE", "text": item})

        snils_patterns = [ re.compile(r"\b\d{3}-\d{3}-\d{3}\s?\d{2}\b"), re.compile(r"\b\d{11}\b") ]
        for rx in snils_patterns:
            for m in rx.finditer(text):
                span = m.group()
                if is_valid_snils(span) and span.lower() not in found_spans:
                    found_other_pii.append({"type": "RUS_SNILS", "text": span})
                    found_spans.add(span.lower())

        phone_rx = re.compile(r"(?:\+7|8)\s?\(?\d{3}\)?\s?\d{3}-?\d{2}-?\d{2}")
        for m in phone_rx.finditer(text):
            raw = m.group()
            if raw.lower() not in found_spans:
                norm = re.sub(r'[^\d]', '', raw)
                if norm.startswith("8"): norm = "7" + norm[1:]
                found_other_pii.append({"type": "PHONE", "text": "+" + norm})
                found_spans.add(raw.lower())

        email_results = analyzer.analyze(text=text, entities=["EMAIL_ADDRESS"], language="en")
        for e in email_results:
            span = text[e.start:e.end]
            if span.lower() not in found_spans:
                found_other_pii.append({"type": "EMAIL_ADDRESS", "text": span})
                found_spans.add(span.lower())
        
        passport_pattern = re.compile(r"\b\d{4}\s?\d{6}\b")
        for m in passport_pattern.finditer(text):
            span = m.group()
            if span.lower() not in found_spans:
                found_other_pii.append({"type": "RUS_PASSPORT", "text": span})
                found_spans.add(span.lower())

        inn_pattern = re.compile(r"\b\d{12}\b")
        for m in inn_pattern.finditer(text):
             span = m.group()
             if span.lower() not in found_spans: 
                found_other_pii.append({"type": "RUS_INN", "text": span})
                found_spans.add(span.lower())

        if found_persons and found_other_pii:
            results.extend(found_persons)
            results.extend(found_other_pii)
            results.append({"type": "PII_COMBO", "text": "Обнаружена комбинация ФИО и других персональных данных"})

    else:
        pii = analyzer.analyze(text=text, language='en')
        for e in pii:
            results.append({ "type": e.entity_type, "text": text[e.start:e.end] })
            
    return results

def moderate_text(text: str, lang: str):
    pii_results = extract_pii(text, lang)
    has_pii = len(pii_results) > 0

    inputs = tox_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = tox_model(**inputs)
        probs = outputs.logits.softmax(dim=1)[0]
        toxic_score = float(probs[1])
    is_toxic = toxic_score >= 0.1 

    return has_pii, is_toxic, pii_results