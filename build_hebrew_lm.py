"""Rebuild models/hebrew-lm/words.tsv (Hebrew word frequencies for the word model) from wordfreq.

    pip install wordfreq
    python build_hebrew_lm.py          # then: python tune_hebrew_lm.py

wordfreq's Hebrew "large" list blends Wikipedia, news, web text and subtitles (data CC BY-SA 4.0).
Deliberately NOT used: the handwriting datasets' own transcriptions - many writers copied the same
texts, so they would make the test words "known" and inflate every measurement.
Keeps meta.json settings (lm_weight, word_bonus, prefix_logp) and course_words.txt as they are.
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wordfreq  # noqa: E402

from recog.hebrew import WordLM  # noqa: E402

out = os.path.join(HERE, "models", "hebrew-lm")
os.makedirs(out, exist_ok=True)
freq = {}
for w in wordfreq.top_n_list("he", 10 ** 6, wordlist="large"):
    k = WordLM.key(w)                                   # Hebrew letters only, as the recognizer looks words up
    if k:
        freq[k] = freq.get(k, 0.0) + wordfreq.word_frequency(w, "he", wordlist="large")
total = sum(freq.values())
items = sorted(freq.items(), key=lambda kv: -kv[1])
with open(os.path.join(out, "words.tsv"), "w", encoding="utf-8") as f:
    for w, p in items:
        f.write(f"{w}\t{math.log(p / total):.4f}\n")
meta_p = os.path.join(out, "meta.json")
meta = json.load(open(meta_p, encoding="utf-8")) if os.path.isfile(meta_p) else {"prefix_logp": -4.5}
meta.update({"oov_logp": round(math.log(items[-1][1] / total) - math.log(10), 4),   # 1 Zipf unit below the rarest
             "words": len(items),
             "sources": ["wordfreq Hebrew 'large' list (Wikipedia, subtitles, news, web, social media; data CC BY-SA 4.0)"]})
json.dump(meta, open(meta_p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"wrote {len(items)} words to {os.path.relpath(out, HERE)}/words.tsv - re-run tune_hebrew_lm.py")
