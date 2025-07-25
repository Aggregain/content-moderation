import re
from app.deps import nlp_ru, analyzer, pii_patterns
from app.deps import tox_model, tox_tokenizer
import torch


def extract_pii(text: str, lang: str):
    final_results = []
    if lang == 'ru':
        found_persons = []
        found_snils = []
        found_other_pii = []
        found_spans = set()

        doc = nlp_ru(text)
        entities = {ent.type: [] for sent in doc.sentences for ent in sent.ents}
        for sent in doc.sentences:
            for ent in sent.ents:
                if ent.text.lower() not in [e.lower() for e in entities[ent.type]]:
                    entities[ent.type].append(ent.text)

        if "PER" in entities:
            found_persons.extend([{"type": "PERSON", "text": item} for item in entities["PER"]])

    
        address_keywords = ['адрес', 'проживает', 'улица', 'ул.', 'дом', 'квартира', 'кв.', 'проспект']
        if any(kw in text.lower() for kw in address_keywords):
       
            address_parts = entities.get("LOC", []) + entities.get("GPE", [])
            if address_parts:
                full_address = ", ".join(address_parts)
                if full_address.lower() not in found_spans:
                    found_other_pii.append({"type": "ADDRESS", "text": full_address})
                    found_spans.add(full_address.lower())
        
    
        birth_keywords = ['родился в', 'родилась в', 'родом из', 'в городе']
        for keyword in birth_keywords:
            match = re.search(rf'{re.escape(keyword)}\s+((?:[А-ЯЁ][а-яё]+(?:[-\s]?[А-ЯЁа-яё]+)*))', text, re.IGNORECASE)
            if match:
                place_name = match.group(1)
                if place_name.lower() not in found_spans:
                    found_other_pii.append({"type": "BIRTH_PLACE", "text": place_name})
                    found_spans.add(place_name.lower())
        
  
        for pattern_info in pii_patterns:
            pii_type = pattern_info['entity_type']
            regex = pattern_info['regex']
            rx = re.compile(regex)
            
            for m in rx.finditer(text):
                span = m.group()
                normalized_span = ''.join(filter(str.isdigit, span))
                if normalized_span not in found_spans:
                    if pii_type == "RUS_SNILS":
                        found_snils.append({"type": pii_type, "text": span})
                    else:
                        found_other_pii.append({"type": pii_type, "text": span})
                    found_spans.add(normalized_span)
        
   
        email_results = analyzer.analyze(text=text, entities=["EMAIL_ADDRESS"], language="en")
        for e in email_results:
            span = text[e.start:e.end]
            if span.lower() not in found_spans:
                 found_other_pii.append({"type": "EMAIL_ADDRESS", "text": span})
                 found_spans.add(span.lower())
      
        if found_snils:
            final_results.extend(found_snils)
            if found_persons:
                final_results.extend(found_persons)
        elif found_persons and found_other_pii:
            final_results.extend(found_persons)
            final_results.extend(found_other_pii)

    else:
        pii = analyzer.analyze(text=text, language='en')
        final_results.extend([{"type": e.entity_type, "text": text[e.start:e.end]} for e in pii])
            
    return final_results

def moderate_text(text: str, lang: str):
    pii_results = extract_pii(text, lang)
    has_pii = len(pii_results) > 0
    is_toxic = False
    if text:
        inputs = tox_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = tox_model(**inputs)
            probs = outputs.logits.softmax(dim=1)[0]
            toxic_score = float(probs[1])
            is_toxic = toxic_score >= 0.1
    return has_pii, is_toxic, pii_results