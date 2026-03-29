# Model Card: Mood Machine

This model card is for the Mood Machine project, which includes **two** versions of a mood classifier:

1. A **rule based model** implemented in `mood_analyzer.py`
2. A **machine learning model** implemented in `ml_experiments.py` using scikit learn

You may complete this model card for whichever version you used, or compare both if you explored them.

## 1. Model Overview

**Model type:**
I implemented and evaluated a rule-based model in `mood_analyzer.py`. I also explored a machine learning model in `ml_experiments.py` using scikit-learn for comparison.

**Intended purpose:**
The system classifies short, informal text messages (social media posts, chat messages) into one of four mood categories: `positive`, `negative`, `neutral`, or `mixed`. It is designed for short-form, colloquial English including slang and emojis.

**How it works (brief):**
The rule-based model computes a sentiment score using handcrafted linguistic rules:

- Positive and negative words from `dataset.py` contribute +1 or -1 respectively
- Slang words (e.g., `"lol"` → +1, `"mid"` → -2, `"wild"` → +1) add direct sentiment signals
- Emojis contribute sentiment signals (e.g., `😂` → +2, `💀` → -1, `😒` → -2, `🔥` → +2)
- Intensifiers (`"very"`, `"so"`, `"really"`, `"super"` → ×2; `"extremely"` → ×3) amplify the surrounding token's score
- Downtoners (`"kind"`, `"kinda"`, `"slightly"`, `"little"` → ×-1) reduce sentiment strength
- Negation is handled with a scoped window of 2 tokens (e.g., `"not happy"` → negative, `"not bad"` → +2 via a phrase override)
- Phrase-level overrides handle common patterns like `"not bad"` (+2) and `"not happy"` (-2)

The final score maps to labels using thresholds:
- `score >= 1` → `"positive"`
- `score <= -1` → `"negative"`
- `-1 < score < 1` (i.e., `score == 0`) → `"neutral"`

The ML model uses a Bag-of-Words representation (CountVectorizer) fed into a Logistic Regression classifier trained on `SAMPLE_POSTS` and `TRUE_LABELS`. It learns statistical word-to-label associations from the data rather than applying handcrafted rules.


## 2. Data

**Dataset description:**
The dataset uses `SAMPLE_POSTS`, which contains 16 labeled short English posts total. 6 came from the starter code, and 10 were added to expand coverage of informal language styles including slang, emojis, sarcasm, and mixed emotions.

**Labeling process:**
Labels in `TRUE_LABELS` were assigned manually based on perceived human sentiment. Some posts were genuinely difficult to label:

- `"Feeling tired but kind of hopeful"` — labeled `mixed` because it contains both a negative word (`tired`) and a positive emotion (`hopeful`), but the `but` pivot makes the overall tone ambiguous.
- `"That was wild lol"` — labeled `neutral` because `"wild"` and `"lol"` are tonally ambiguous slang; the meaning depends heavily on context not present in the sentence.
- `"This is fine 🔥"` — labeled `mixed` because the phrase `"This is fine"` is a well-known internet meme implying understated distress, while `🔥` is a positive/hype emoji, creating genuine ambiguity.

**Important characteristics of your dataset:**
- Contains emojis: `😂`, `🔥`, `💀`, `😒`
- Contains slang and informal speech: `"lowkey"`, `"mid"`, `"lol"`, `"bruh"`, `"wild"`
- Includes mixed emotional statements (conflicting words in the same sentence)
- Contains short, context-dependent sentences where meaning depends on shared cultural knowledge (e.g., sarcasm, memes)

**Possible issues with the dataset:**
- Very small dataset size (16 examples) severely limits generalization
- Label subjectivity: two people may reasonably disagree on `"mixed"` vs `"neutral"` for several posts
- Class imbalance: `"mixed"` and `"neutral"` appear less frequently than `"positive"` and `"negative"`, which may bias both models


## 3. How the Rule Based Model Works (if used)

**Your scoring rules:**
- **Base word scoring:** each token is checked against `self.positive_words` (+1) and `self.negative_words` (-1), both sourced from `dataset.py`.
- **Slang scoring:** a separate `slang_pos` and `slang_neg` dictionary adds scores for informal words that the base word lists miss (e.g., `"mid"` → -2, `"lol"` → +1).
- **Emoji scoring:** an `emoji_map` assigns sentiment values to common emojis (e.g., `😂` → +2, `😒` → -2).
- **Intensifiers:** words like `"very"` and `"so"` multiply the current token's score by 2–3, but only affect the same token they are on (not the next word), so they only help if the intensifier itself has a score.
- **Negation window:** when a negation word (`"not"`, `"no"`, `"never"`, `"n't"`) is found, a counter of 2 is set and the next 2 tokens have their scores flipped.
- **Phrase overrides:** specific two-word phrases like `"not bad"` are caught before individual token scoring and given a fixed score (+2 for `"not bad"`, -2 for `"not happy"`, etc.).
- **Label thresholds:** only scores of exactly 0 map to `"neutral"`; scores ≥1 are `"positive"` and ≤-1 are `"negative"`. The label `"mixed"` is unreachable in `predict_label` because the `else` branch can never trigger given the three conditions cover all integers.

**Strengths of this approach:**
- Transparent and interpretable: the `explain()` method shows exactly which tokens contributed to the score.
- Handles negation reliably for simple two-word phrases (`"not happy"`, `"not bad"`).
- Handles slang and emojis explicitly once added to the lookup dictionaries.
- Predictable behavior on clear-cut positive or negative posts (e.g., `"I love this class"`, `"Today was a terrible day"`).

**Weaknesses of this approach:**
- **Cannot predict `"mixed"`:** the label mapping in `predict_label` has an unreachable `else` branch; any post with a nonzero score becomes strictly positive or negative, and score=0 maps to `"neutral"` not `"mixed"`. Posts with balanced positive and negative signals (score=0) are labeled `"neutral"` instead of `"mixed"`.
- **Sarcasm is invisible:** `"I absolutely love waiting in long lines 😒"` is predicted `negative` (correct by accident via the 😒 emoji), but the rule system has no awareness of sarcastic intent — it would predict `positive` without the emoji.
- **Intensifiers misfire:** `"So excited for the weekend"` — `"so"` contributes 0 score because its own token_score is 0 before the intensifier multiplier applies. Intensifiers only amplify themselves, not the next token.
- **Context-free:** each word is scored in isolation. The system has no concept of discourse structure, topic, or sentence-level meaning.
- **Unknown words score 0:** `"hopeful"`, `"lowkey"`, `"excited"` (wait — `excited` IS in positive_words) — any word not in the hardcoded lists is silently ignored.


## 4. How the ML Model Works (if used)

**Features used:**
Bag-of-Words representation using scikit-learn's `CountVectorizer`. Each sentence becomes a vector of word counts over the vocabulary of the training set.

**Training data:**
The model trains on `SAMPLE_POSTS` (16 examples) and `TRUE_LABELS` from `dataset.py`.

**Training behavior:**
Adding more labeled examples generally improved the model's ability to assign the right label class, but with only 16 examples the model is prone to memorizing the training data. Changing labels on ambiguous examples (e.g., `"neutral"` vs `"mixed"`) visibly shifted predictions on similar posts, showing how sensitive the model is to the small dataset.

**Strengths and weaknesses:**
- **Strengths:** learns patterns automatically without handcrafting rules; can pick up on informal words and phrases once they appear in training data.
- **Weaknesses:** with 16 training examples it severely overfits; accuracy on training data is misleadingly high. It cannot generalize to words it has never seen. It also has no understanding of negation, sarcasm, or word order — `"not happy"` and `"happy not"` would be treated identically.


## 5. Evaluation

**How you evaluated the model:**
Both models were evaluated on the full `SAMPLE_POSTS` list using `evaluate_rule_based()` in `main.py`, which compares `predict_label()` output against `TRUE_LABELS` and computes accuracy.

**Estimated accuracy (rule-based):** approximately **10/16 (63%)** based on tracing the scoring logic against each post.

**Examples of correct predictions:**

| Post | Predicted | True | Why correct |
|---|---|---|---|
| `"I love this class so much"` | positive | positive | `"love"` is in `positive_words` (+1); score=1 → positive |
| `"I am not happy about this"` | negative | negative | Negation window flips `"happy"` from +1 to -1; score=-1 → negative |
| `"This food is mid 💀"` | negative | negative | `"mid"` (slang_neg: -2) + `💀` (emoji_map: -1) = -3 → negative |

**Examples of incorrect predictions:**

| Post | Predicted | True | Why wrong |
|---|---|---|---|
| `"Feeling tired but kind of hopeful"` | negative | mixed | `"tired"` scores -1; `"hopeful"` is not in any word list so scores 0; `"but"` has no effect on scoring; score=-1 → negative instead of mixed |
| `"That was wild lol"` | positive | neutral | `"wild"` (+1) and `"lol"` (+1) in `slang_pos` sum to +2 → positive, but both words are tonally neutral in context |
| `"This is fine 🔥"` | positive | mixed | `🔥` emoji scores +2 → positive, but the post is culturally ambiguous (meme reference implying hidden distress) |


## 6. Limitations

- **Dataset is very small (16 examples):** both models have seen almost every possible example during training; accuracy on unseen text would be much lower.
- **`"mixed"` label is unreachable in the rule-based model:** the `predict_label` thresholds never return `"mixed"`, so the model systematically misclassifies all truly mixed-sentiment posts.
- **Intensifiers apply to themselves, not the next token:** `"so excited"` does not get amplified because `"so"` has no base score of its own to multiply.
- **No sarcasm detection:** the model cannot distinguish `"I absolutely love waiting in long lines 😒"` as sarcastic without relying on the emoji as a proxy signal.
- **No generalization to longer text:** the word-lookup approach does not account for discourse structure, topic shifts, or compound sentences beyond a two-token negation window.
- **Cultural and community-specific language:** slang and meme references (e.g., `"This is fine 🔥"`) require shared cultural knowledge that neither a word list nor a small ML model can reliably learn.


## 7. Ethical Considerations

- **Misclassifying distress:** a message expressing genuine distress with ironic or understated language (e.g., `"I'm fine 🙂"`) could be classified as `positive`, potentially causing harm if this system were used in a mental health context.
- **Language community bias:** the word lists and training data reflect one particular style of informal English. Slang from other communities, languages, or dialects would score 0 (ignored) and likely produce `"neutral"` predictions regardless of actual sentiment.
- **Privacy:** analyzing personal messages for mood — even for benign purposes — raises privacy concerns. Users may not consent to having their emotional state inferred and stored.
- **False confidence:** the model returns a single label with no confidence score or uncertainty signal, which could mislead downstream uses (e.g., flagging posts) into treating uncertain predictions as reliable decisions.


## 8. Ideas for Improvement

- **Fix the `"mixed"` label:** adjust `predict_label` thresholds so that posts with both positive and negative signals (e.g., score near 0 but with hits on both sides) return `"mixed"` rather than `"neutral"`.
- **Fix intensifier scoping:** modify `score_text` so intensifiers amplify the *next* token's score rather than their own, which is the linguistically correct behavior.
- **Add more labeled data:** even doubling to 30–40 diverse examples would meaningfully improve the ML model's generalization.
- **Use TF-IDF instead of CountVectorizer:** would downweight very common words and better surface discriminative terms.
- **Add a real held-out test set:** currently both models evaluate on the same data they were designed or trained for; a separate test set would give a honest accuracy estimate.
- **Improve emoji and slang coverage:** expand `emoji_map` and slang dictionaries, or use an external emoji sentiment lexicon.
- **Add sarcasm signals:** detect common sarcasm markers (e.g., an intensely positive word followed immediately by a negative emoji) and apply a score reversal.
- **Use a small pretrained model:** a lightweight transformer (e.g., DistilBERT fine-tuned on sentiment) would handle context, negation, and sarcasm far more robustly than either current approach.
