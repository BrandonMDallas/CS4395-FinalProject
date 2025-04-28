import re
import spacy
import spacy.cli


class TextPreprocessor:
    def __init__(
        self,
        model_name: str = "en_core_web_trf",
        disable: list[str] = ["parser", "ner"],
        remove_stopwords: bool = False,
        remove_punct: bool = True,
    ):
        """
        Initialize the text preprocessor.

        Args:
            model_name:      SpaCy model to load (e.g., transformer or small).
            disable:         List of pipeline components to disable for efficiency.
            remove_stopwords: Whether to drop stopwords from the output.
            remove_punct:    Whether to drop punctuation tokens from the output.
        """
        self.remove_stopwords = remove_stopwords
        self.remove_punct = remove_punct

        # Try loading the model; if missing, download it and load again
        try:
            self.nlp = spacy.load(model_name, disable=disable)
        except OSError:
            spacy.cli.download(model_name)  # fetches the model
            self.nlp = spacy.load(model_name, disable=disable)

        # Pre­compile regexes
        self._rx_num_letter = re.compile(r"(\d+)([A-Za-z])")
        self._rx_letter_num = re.compile(r"([A-Za-z])(\d+)")
        self._rx_space = re.compile(r"\s+")

    def preprocess(self, text: str) -> str:
        """
        Clean and tokenize a piece of text.

        Steps:
          1) Regex-based normalization for alphanumeric separation and whitespace.
          2) SpaCy tokenization + lemmatization.
          3) Optional filtering of punctuation and stopwords.
          4) Lowercasing and joining tokens into a final string.

        Args:
            text: Raw input string.

        Returns:
            A cleaned, lemmatized, lowercase string of tokens.
        """
        text = self._rx_num_letter.sub(r"\1 \2", text)
        text = self._rx_letter_num.sub(r"\1 \2", text)
        text = self._rx_space.sub(" ", text)

        doc = self.nlp(text)
        tokens = []
        for tok in doc:
            if self.remove_punct and tok.is_punct:
                continue
            if self.remove_stopwords and tok.is_stop:
                continue
            if tok.is_alpha or tok.is_digit:
                tokens.append(tok.lemma_.lower())
        return " ".join(tokens)
