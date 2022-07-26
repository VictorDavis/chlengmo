# bloody dependencies
from collections import Counter
import math
import random

# internal dependencies
from .exceptions import ModelNotFittedError


class Chlengmo:
    def __init__(self, n: int):
        """
        Initialize a Character-Level N-Gram Model.

        :param int n: N-Gram length.
        """

        self.n = abs(int(n))

    def fit(self, text: str):
        """
        Fit model to text.

        :param str text: String to fit model to.
        """

        # calculate alphabet and char <-> int lookup data structures
        alphabet = Counter(text).most_common()
        self._int2char = [char for char, count in alphabet]
        self._char2int = {char: idx for idx, char in enumerate(self._int2char)}
        self._base = len(alphabet)
        self._maxinput = max(1, self._base ** (self.n - 1))

        # special case: n=0 -> "0-gram" uniform distribution
        if self.n == 0:
            self._model = {0: [1] * self._base}
            return self

        # special case: n=1 -> "1-gram" empirical character-level distribution
        if self.n == 1:
            self._model = {0: [count for char, count in alphabet]}
            return self

        # build ngram model data structure
        input = 0
        self._model = {}
        for char in text:

            # output
            output = self._char2int[char]

            # initialize counter for new inputs
            if input not in self._model:
                self._model[input] = [0] * self._base

            # update counter
            self._model[input][output] += 1

            # update input
            input = (self._base * input + output) % self._maxinput

        # return self, for method chaining
        return self

    @property
    def is_fitted(self):
        attributes = [
            "_int2char",
            "_char2int",
            "_base",
            "_maxinput",
            "_model",
        ]
        return all(hasattr(self, attr) for attr in attributes)

    @property
    def entropy(self):

        # REF: https://www.princeton.edu/~wbialek/rome/refs/shannon_51.pdf

        # n-gram entropy
        ngram_counts = [
            count for counts in self._model.values() for count in counts if count > 0
        ]
        ngram_total = sum(ngram_counts)
        ngram_probs = [count / ngram_total for count in ngram_counts]
        ngram_entropy = -sum([p * math.log2(p) for p in ngram_probs])

        # (n-1)-gram entropy
        prefix_counts = [sum(counts) for counts in self._model.values()]
        prefix_total = sum(prefix_counts)
        prefix_probs = [count / prefix_total for count in prefix_counts]
        prefix_entropy = -sum([p * math.log2(p) for p in prefix_probs])

        # difference
        entropy = ngram_entropy - prefix_entropy
        return entropy

    def generate(self, length: int, prompt: str = "", seed: int = None) -> str:
        """
        Generate fake text.

        :param int length: Length of fake text to generate (in characters).
        :param str prompt: Prompt the fake text with a starting string.
        :param int seed: For pseudo-random number generator, for replicating results.
        :raise: ModelNotFittedError if model hasn't been fitted yet
        """

        # make sure model is fitted
        if not self.is_fitted:
            raise ModelNotFittedError("Model hasn't been fitted yet!")

        # prepare input
        input = 0
        for char in prompt:
            output = self._char2int[char]
            input = (self._base * input + output) % self._maxinput

        # set seed
        random.seed(seed)

        # generate text
        text = prompt
        for _ in range(length):
            if input in self._model:
                weights = self._model[input]
                outputs = random.choices(population=range(self._base), weights=weights)
                output = outputs[0]
            else:
                output = 0
            text += self._int2char[output]

            # update input
            input = (self._base * input + output) % self._maxinput

        # return generated text
        return text
