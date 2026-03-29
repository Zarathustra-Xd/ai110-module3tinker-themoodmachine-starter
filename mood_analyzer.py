# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""
import re
from typing import List, Dict, Tuple, Optional

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        TODO: Improve this method.

        Right now, it does the minimum:
          - Strips leading and trailing whitespace
          - Converts everything to lowercase
          - Splits on spaces

        Ideas to improve:
          - Remove punctuation
          - Handle simple emojis separately (":)", ":-(", "🥲", "😂")
          - Normalize repeated characters ("soooo" -> "soo")
        """
        cleaned = text.strip().lower()
    
        # Remove punctuation (keep emojis)
        cleaned = re.sub(r"[^\w\s:)(😂💀😒🥲🔥]", "", cleaned) 
    
        tokens = cleaned.split()

        return tokens

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" for the given text.

        Positive words increase the score.
        Negative words decrease the score.

        TODO: You must choose AT LEAST ONE modeling improvement to implement.
        For example:
          - Handle simple negation such as "not happy" or "not bad"
          - Count how many times each word appears instead of just presence
          - Give some words higher weights than others (for example "hate" < "annoyed")
          - Treat emojis or slang (":)", "lol", "💀") as strong signals
        """
        # TODO: Implement this method.
        #   1. Call self.preprocess(text) to get tokens.
        #   2. Loop over the tokens.
        #   3. Increase the score for positive words, decrease for negative words.
        #   4. Return the total score.
        #
        # Hint: if you implement negation, you may want to look at pairs of tokens,
        # like ("not", "happy") or ("never", "fun").
        
        tokens = self.preprocess(text)
        score = 0

        negation_words = {"not", "no", "never", "n't"}
        intensifiers = {"very": 2, "so": 2, "really": 2, "super": 2, "extremely": 3}
        downtoners = {"kind": -1, "kinda": -1, "sort": -1, "slightly": -1, "little": -1}

        emoji_map = {
            "😂": 2, "🔥": 2, "😊": 1,
            "😒": -2, "💀": -1, "🥲": -1
        }

        slang_pos = {"lol": 1, "haha": 1, "wild": 1, "lit": 2, "good": 1}
        slang_neg = {"mid": -2, "trash": -3, "annoying": -2, "tired": -1}

        neg_window = 0  # FIX: proper scoped negation instead of global flip

        i = 0
        while i < len(tokens):
            token = tokens[i]

            # ----------------------------
            # 1. NEGATION SCOPING FIX
            # ----------------------------
            if token in negation_words:
                neg_window = 2  # affects next 2 tokens only
                i += 1
                continue

            is_negated = neg_window > 0

            # ----------------------------
            # 2. PHRASE HANDLING (context fix)
            # ----------------------------
            if i + 1 < len(tokens):
                phrase = token + " " + tokens[i + 1]

                if phrase == "not bad":
                    score += 2
                    i += 2
                    neg_window = max(0, neg_window - 1)
                    continue

                if phrase in {"not good", "not happy", "not great"}:
                    score -= 2
                    i += 2
                    neg_window = max(0, neg_window - 1)
                    continue

                if phrase in {"kind of good", "kinda good"}:
                    score += 0
                    i += 2
                    neg_window = max(0, neg_window - 1)
                    continue

            # ----------------------------
            # 3. BASE SENTIMENT
            # ----------------------------
            token_score = 0

            if token in self.positive_words:
                token_score += 1
            elif token in self.negative_words:
                token_score -= 1

            # ----------------------------
            # 4. SLANG + CONTEXT FIX
            # ----------------------------
            if token in slang_pos:
                token_score += slang_pos[token]
            if token in slang_neg:
                token_score += slang_neg[token]

            # ----------------------------
            # 5. EMOJI (kept moderate)
            # ----------------------------
            if token in emoji_map:
                token_score += emoji_map[token]

            # ----------------------------
            # 6. INTENSIFIERS / DOWNTONERS
            # ----------------------------
            if token in intensifiers:
                token_score *= intensifiers[token]

            if token in downtoners:
                token_score *= downtoners[token]

            # ----------------------------
            # 7. NEGATION (LOCAL FIX)
            # ----------------------------
            if is_negated:
                token_score *= -1
                neg_window -= 1

            score += token_score
            i += 1

        return score

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the numeric score for a piece of text into a mood label.

        The default mapping is:
          - score > 0  -> "positive"
          - score < 0  -> "negative"
          - score == 0 -> "neutral"

        TODO: You can adjust this mapping if it makes sense for your model.
        For example:
          - Use different thresholds (for example score >= 2 to be "positive")
          - Add a "mixed" label for scores close to zero
        Just remember that whatever labels you return should match the labels
        you use in TRUE_LABELS in dataset.py if you care about accuracy.
        """
        # TODO: Implement this method.
        #   1. Call self.score_text(text) to get the numeric score.
        #   2. Return "positive" if the score is above 0.
        #   3. Return "negative" if the score is below 0.
        #   4. Return "neutral" otherwise.
        score = self.score_text(text)

        if score >= 1:
            return "positive"
        elif score <= -1:
            return "negative"
        elif -1 < score < 1:
            return "neutral"
        else:
            return "mixed"

    # ---------------------------------------------------------------------
    # Explanations (optional but recommended)
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.

        TODO:
          - Look at the tokens and identify which ones counted as positive
            and which ones counted as negative.
          - Show the final score.
          - Return a short human readable explanation.

        Example explanation (your exact wording can be different):
          'Score = 2 (positive words: ["love", "great"]; negative words: [])'

        The current implementation is a placeholder so the code runs even
        before you implement it.
        """
        tokens = self.preprocess(text)

        positive_hits = []
        negative_hits = []
        score = 0

        i = 0
        while i < len(tokens):
            token = tokens[i]

            if token == "but":
                score = 0
                i += 1
                continue

            if token in ["not", "no", "never"] and i + 1 < len(tokens):
                next_token = tokens[i + 1]

                if next_token in self.positive_words:
                    negative_hits.append(f"not {next_token}")
                    score -= 1
                    i += 2
                    continue
                elif next_token in self.negative_words:
                    positive_hits.append(f"not {next_token}")
                    score += 1
                    i += 2
                    continue

            if token in self.positive_words:
                positive_hits.append(token)
                score += 1
            elif token in self.negative_words:
                negative_hits.append(token)
                score -= 1

            if token in ["😂", "🔥", "😊"]:
                positive_hits.append(token)
                score += 1
            elif token in ["😒", "💀", "🥲"]:
                negative_hits.append(token)
                score -= 1

            if token == "love":
                positive_hits.append(token)
                score += 1
            elif token == "hate":
                negative_hits.append(token)
                score -= 1

            i += 1

        return f"Score = {score} (positive: {positive_hits}, negative: {negative_hits})"
