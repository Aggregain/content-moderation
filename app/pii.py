import re
from app.deps import nlp_ru, is_valid_snils, region_map, analyzer, tox_model, tox_tokenizer
import torch

def extract_pii(text: str, lang: str):

    final_results = []
    
    if lang == 'ru':

        found_persons = []
        found_other_pii = []
        found_spans = set()

        doc = nlp_ru(text)

        entities = {}
        for sent in doc.sentences:
            for ent in sent.ents:
                span = text[ent.start_char:ent.end_char]
                if ent.type not in entities: entities[ent.type] = []
                if span.lower() not in found_spans:
                    entities[ent.type].append(span)
                    found_spans.add(span.lower())
   
        if "PER" in entities:
            for item in entities["PER"]:
                found_persons.append({"type": "PERSON", "text": item})
        
 
        if "GPE" in entities:
            for item in entities["GPE"]:
  
                if not item.upper().startswith(("СНИЛС", "ПАСПОРТ")):
                    found_other_pii.append({"type": "BIRTH_PLACE", "text": item})

        address_keywords = ['улица', 'ул', 'проспект', 'пр-т', 'площадь', 'пл', 'дом', 'квартира', 'кв', 'корпус', 'строение', 'город', 'поселок', 'деревня']
        address_pattern = re.compile(r'\b(' + '|'.join(address_keywords) + r')\b', re.IGNORECASE)
        if address_pattern.search(text) and "LOC" in entities:
            full_address = ", ".join(entities["LOC"])
            found_other_pii.append({"type": "ADDRESS", "text": full_address})
            

        pii_patterns = {
            "RUS_PASSPORT": [re.compile(r"\b\d{4}\s?\d{6}\b")],
            "RUS_INN":      [re.compile(r"\b\d{12}\b")],
            "RUS_SNILS":    [re.compile(r"\b\d{3}-\d{3}-\d{3}\s?\d{2}\b"), re.compile(r"\b\d{11}\b")],
            "PHONE":        [re.compile(r"(?:\+7|8)\s?\(?\d{3}\)?\s?\d{3}-?\d{2}-?\d{2}")]
        }

        for pii_type, patterns in pii_patterns.items():
            for rx in patterns:
                for m in rx.finditer(text):
                    span = m.group()
                    if span.lower() not in found_spans:

                        if pii_type == "RUS_SNILS" and not is_valid_snils(span):
                            continue
                        found_other_pii.append({"type": pii_type, "text": span})
                        found_spans.add(span.lower())

        email_results = analyzer.analyze(text=text, entities=["EMAIL_ADDRESS"], language="en")
        for e in email_results:
            span = text[e.start:e.end]
            if span.lower() not in found_spans:
                found_other_pii.append({"type": "EMAIL_ADDRESS", "text": span})
                found_spans.add(span.lower())


        final_results.extend(found_other_pii)

        if found_persons and found_other_pii:
            final_results.extend(found_persons)


    else:
        pii = analyzer.analyze(text=text, language='en')
        final_results.extend([{ "type": e.entity_type, "text": text[e.start:e.end] } for e in pii])
            
    return final_results

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