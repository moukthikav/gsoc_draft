import wikipedia
import json

wikipedia.set_lang("hi")

topics = [
    "भारत",
    "दिल्ली",
    "सचिन तेंदुलकर",
    "गंगा नदी",
    "ताज महल"
]

sentences = []

for topic in topics:
    page = wikipedia.page(topic)

    text = page.content
    sents = text.split("।")

    for s in sents:
        s = s.strip()
        if len(s) > 20:
            sentences.append(s)

with open("data/wiki_sentences.json", "w", encoding="utf-8") as f:
    json.dump(sentences[:200], f, ensure_ascii=False, indent=2)

print("Extracted", len(sentences), "sentences")