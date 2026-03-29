"""
combined_labeler.py

Auto-labels Hindi Wikipedia sentences using:
1. Expanded rule-based patterns
2. Existing gold triples (carry forward as-is)
3. Deduplication and conflict flagging

Output: data/combined_candidates.json  — ready for human review
        data/auto_training_data.json   — approved triples in training format
"""

import json
import re
import os

# ─────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────

with open("data/wiki_sentences.json", "r", encoding="utf-8") as f:
    wiki_sentences = json.load(f)

with open("data/gold_triples.json", "r", encoding="utf-8") as f:
    gold_triples = json.load(f)

print(f"📦 Loaded {len(wiki_sentences)} wiki sentences")
print(f"📦 Loaded {len(gold_triples)} gold triples")

# ─────────────────────────────────────────────
# 2. Expanded Rule-Based Extractor
# ─────────────────────────────────────────────

def extract_by_rules(sentence):
    """Returns list of (subject, predicate, object, rule_name) tuples."""
    results = []
    s = sentence.strip().rstrip("।.")

    # Rule 1: X की राजधानी Y है / X Y की राजधानी है
    if "की राजधानी" in s:
        # Pattern: "Y X की राजधानी है" → subject=Y, pred=capital, obj=X
        m = re.search(r'(.+?)\s+(.+?)की राजधानी', s)
        if m:
            subj = m.group(1).strip()
            obj  = m.group(2).strip()
            results.append((subj, "dbo:capital", obj, "राजधानी"))

    # Rule 2: X में स्थित है / X Y में स्थित है
    if "में स्थित" in s:
        m = re.search(r'(.+?)\s+(.+?)में स्थित', s)
        if m:
            subj = m.group(1).strip()
            obj  = m.group(2).strip()
            results.append((subj, "dbo:location", obj, "स्थित"))

    # Rule 3: X का निर्माण Y ने किया
    if "का निर्माण" in s and "ने" in s:
        m = re.search(r'(.+?)का निर्माण\s+(.+?)ने', s)
        if m:
            subj = m.group(1).strip()
            obj  = m.group(2).strip()
            results.append((subj, "dbo:builder", obj, "निर्माण"))

    # Rule 4: X Y का उत्कृष्ट नमूना है (X is example of Y)
    if "का उत्कृष्ट नमूना" in s:
        m = re.search(r'(.+?)\s+(.+?)का उत्कृष्ट नमूना', s)
        if m:
            subj = m.group(1).strip()
            obj  = m.group(2).strip()
            results.append((subj, "dbo:style", obj, "नमूना"))

    # Rule 5: X युनेस्को विश्व धरोहर स्थल बना
    if "युनेस्को विश्व धरोहर" in s:
        m = re.search(r'(.+?)\s+युनेस्को विश्व धरोहर', s)
        if m:
            subj = m.group(1).strip()
            results.append((subj, "dbo:heritageStatus", "UNESCO_World_Heritage_Site", "यूनेस्को"))

    # Rule 6: X के Y राज्य की राजधानी है
    if "राज्य की राजधानी" in s:
        m = re.search(r'(.+?)\s+(.+?)राज्य की राजधानी', s)
        if m:
            subj = m.group(1).strip()
            obj  = m.group(2).strip()
            results.append((subj, "dbo:capital", obj, "राज्य_राजधानी"))

    # Rule 7: X नदी के किनारे स्थित
    if "नदी के किनारे" in s:
        m = re.search(r'(.+?)\s+(.+?)नदी के किनारे', s)
        if m:
            subj = m.group(1).strip()
            obj  = m.group(2).strip() + " नदी"
            results.append((subj, "dbo:locatedOnWaterBody", obj, "नदी_किनारे"))

    # Rule 8: X एशिया / भारत में स्थित ... पर्वत / नदी / देश है
    if "एशिया में स्थित" in s:
        m = re.search(r'(.+?)\s+एशिया में स्थित', s)
        if m:
            subj = m.group(1).strip()
            results.append((subj, "dbo:location", "Asia", "एशिया_स्थित"))

    # Rule 9: X का जन्म Y में हुआ
    if "का जन्म" in s and "में हुआ" in s:
        m = re.search(r'(.+?)का जन्म\s+(.+?)में हुआ', s)
        if m:
            subj = m.group(1).strip()
            obj  = m.group(2).strip()
            results.append((subj, "dbo:birthPlace", obj, "जन्म"))

    # Rule 10: X को Y के नाम से जाना जाता है
    if "के नाम से" in s and "जाना जाता" in s:
        m = re.search(r'(.+?)को\s+(.+?)के नाम से', s)
        if m:
            subj = m.group(1).strip()
            obj  = m.group(2).strip()
            results.append((subj, "dbo:alias", obj, "नाम_से_जाना"))

    # Rule 11: X भारत का Y है (X is India's Y)
    if "भारत का" in s and "है" in s:
        m = re.search(r'(.+?)\s+भारत का\s+(.+?)\s+है', s)
        if m:
            subj = m.group(1).strip()
            obj  = m.group(2).strip()
            results.append((subj, "dbo:isA", obj, "भारत_का"))

    # Rule 12: X Y से अलग करता है (separates)
    if "से अलग करता है" in s:
        m = re.search(r'(.+?)\s+(.+?)से अलग करता है', s)
        if m:
            subj = m.group(1).strip()
            obj  = m.group(2).strip()
            results.append((subj, "dbo:separates", obj, "अलग_करता"))

    # Rule 13: X का उद्गम Y है
    if "का उद्गम" in s:
        m = re.search(r'(.+?)का उद्गम\s+(.+?)(?:है|करती)', s)
        if m:
            subj = m.group(1).strip()
            obj  = m.group(2).strip()
            results.append((subj, "dbo:sourceOfRiver", obj, "उद्गम"))

    return results


# ─────────────────────────────────────────────
# 3. Build gold triple lookup (sentence → triples)
# ─────────────────────────────────────────────

gold_lookup = {}
for item in gold_triples:
    gold_lookup[item["sentence"].strip()] = item["triples"]


# ─────────────────────────────────────────────
# 4. Process all 87 wiki sentences
# ─────────────────────────────────────────────

candidates = []
training_ready = []

for item in wiki_sentences:
    sid      = item["id"]
    sentence = item["sentence"].strip()

    entry = {
        "id": sid,
        "sentence": sentence,
        "sources": [],
        "triples": [],
        "status": "pending"   # pending | approved | rejected | conflict
    }

    seen_triples = set()

    # --- Source A: Gold triples (highest trust) ---
    if sentence in gold_lookup:
        for t in gold_lookup[sentence]:
            key = (t["subject"], t["predicate"], t["object"])
            if key not in seen_triples:
                seen_triples.add(key)
                entry["triples"].append({
                    "subject":   t["subject"],
                    "predicate": t["predicate"],
                    "object":    t["object"],
                    "source":    "gold",
                    "confidence": "high"
                })
                entry["sources"].append("gold")

    # --- Source B: Rule-based ---
    rule_hits = extract_by_rules(sentence)
    for (subj, pred, obj, rule_name) in rule_hits:
        # Clean up extracted text
        subj = re.sub(r'[।,]', '', subj).strip()
        obj  = re.sub(r'[।,]', '', obj).strip()

        # Skip very long or empty extractions (likely parsing errors)
        if not subj or not obj or len(subj) > 60 or len(obj) > 60:
            continue

        key = (subj, pred, obj)
        if key not in seen_triples:
            seen_triples.add(key)
            entry["triples"].append({
                "subject":   subj,
                "predicate": pred,
                "object":    obj,
                "source":    f"rule:{rule_name}",
                "confidence": "medium"
            })
            if "rule" not in entry["sources"]:
                entry["sources"].append("rule")

    # Set status
    if not entry["triples"]:
        entry["status"] = "no_extraction"
    elif any(t["source"] == "gold" for t in entry["triples"]):
        entry["status"] = "approved"
    elif len(entry["triples"]) > 1:
        entry["status"] = "conflict_review"   # multiple rule hits, needs human check
    else:
        entry["status"] = "rule_review"       # single rule hit, needs human check

    candidates.append(entry)

    # Auto-approve gold triples into training data immediately
    for t in entry["triples"]:
        if t["source"] == "gold":
            training_ready.append({
                "input":  sentence,
                "output": f"{t['subject']} | {t['predicate']} | {t['object']}"
            })


# ─────────────────────────────────────────────
# 5. Save outputs
# ─────────────────────────────────────────────

os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)

with open("data/combined_candidates.json", "w", encoding="utf-8") as f:
    json.dump(candidates, f, ensure_ascii=False, indent=2)

with open("data/auto_training_data.json", "w", encoding="utf-8") as f:
    json.dump(training_ready, f, ensure_ascii=False, indent=2)

# ─────────────────────────────────────────────
# 6. Summary report
# ─────────────────────────────────────────────

from collections import Counter
status_counts = Counter(c["status"] for c in candidates)
source_counts = Counter(s for c in candidates for s in c["sources"])

print("\n" + "="*50)
print("📊 EXTRACTION SUMMARY")
print("="*50)
print(f"Total sentences processed : {len(candidates)}")
print(f"\nStatus breakdown:")
for status, count in status_counts.items():
    print(f"  {status:<20} : {count}")
print(f"\nSource breakdown:")
for source, count in source_counts.items():
    print(f"  {source:<20} : {count}")
print(f"\nAuto-approved training samples : {len(training_ready)}")
print(f"\n📁 Outputs saved:")
print(f"  data/combined_candidates.json  → all 87 sentences with extractions + status")
print(f"  data/auto_training_data.json   → {len(training_ready)} ready-to-train gold samples")
print("\n✅ Next step: Review 'rule_review' and 'conflict_review' entries in combined_candidates.json")
print("   Change status to 'approved' for correct ones, then run merge_approved.py")