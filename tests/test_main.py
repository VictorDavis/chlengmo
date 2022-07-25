# bloody dependencies
from nltk.corpus import gutenberg
from nltk.lm.models import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
import time
from unittest import TestCase

# internal dependencies
from chlengmo import Chlengmo
from chlengmo.exceptions import ModelNotFittedError


class ChlengmoTest(TestCase):
    def test_basic(self):

        # retrieve corpus from NLTK
        filename = "melville-moby_dick.txt"
        text = gutenberg.raw(filename)
        start = "Call me Ishmael"
        start_idx = text.index(start)
        text = text[start_idx:]

        # create and fit model
        n = 3
        model = Chlengmo(n=n).fit(text)

        # generate fake text
        length = 999
        prompt = "Call me "
        fake_text = model.generate(length=length, prompt=prompt)
        fake_words = fake_text.split(" ")
        assert len(fake_text) == len(prompt) + length
        assert fake_text.startswith(prompt)
        assert len(set(fake_words)) > 99

    def test_seed(self):

        # retrieve corpus from NLTK
        filename = "melville-moby_dick.txt"
        text = gutenberg.raw(filename)
        start = "Call me Ishmael"
        start_idx = text.index(start)
        text = text[start_idx:]

        # create and fit model
        n = 3
        model = Chlengmo(n=n).fit(text)

        # generate fake text (seeded)
        length = 999
        prompt = "Call me "
        seed = 42
        fake_text1 = model.generate(length=length, prompt=prompt, seed=seed)
        fake_text2 = model.generate(length=length, prompt=prompt, seed=seed)
        assert fake_text1 == fake_text2

        # generate fake text (seedless)
        fake_text1 = model.generate(length=length, prompt=prompt)
        fake_text2 = model.generate(length=length, prompt=prompt)
        assert fake_text1 != fake_text2

    def test_speed(self):

        # retrieve corpus from NLTK
        filename = "melville-moby_dick.txt"
        text = gutenberg.raw(filename)
        start = "Call me Ishmael"
        start_idx = text.index(start)
        text = text[start_idx:]

        # fit chlengmo model
        n = 3
        chlengmo = Chlengmo(n=n)
        time_start = time.process_time()
        chlengmo.fit(text)
        time_end = time.process_time()
        time_elapsed_chlengmo = time_end - time_start

        # fit nltk model
        # REF: https://www.nltk.org/api/nltk.lm.html
        train, vocab = padded_everygram_pipeline(n, [list(text)])
        mle = MLE(n)
        time_start = time.process_time()
        mle.fit(train, vocab)
        time_end = time.process_time()
        time_elapsed_nltk = time_end - time_start

        # chlengmo is >10x faster!
        time_factor = time_elapsed_nltk / time_elapsed_chlengmo
        assert time_factor > 10

        # generate fake text from chlengmo model
        length = 9999
        prompt = "Call me "
        time_start = time.process_time()
        fake_text_chlengmo = chlengmo.generate(length=length, prompt=prompt)
        time_end = time.process_time()
        time_elapsed_chlengmo = time_end - time_start
        fake_words_chlengmo = fake_text_chlengmo.split(" ")
        assert len(set(fake_words_chlengmo)) > 999

        # generate fake text from nltk model
        time_start = time.process_time()
        fake_text_nltk = mle.generate(num_words=length, text_seed=list(prompt))
        time_end = time.process_time()
        time_elapsed_nltk = time_end - time_start
        fake_text_nltk = "".join(fake_text_nltk)
        fake_words_nltk = fake_text_nltk.split(" ")
        assert len(set(fake_words_nltk)) > 999

        # chlengmo is >10x faster!
        time_factor = time_elapsed_nltk / time_elapsed_chlengmo
        assert time_factor > 10

    def test_exceptions(self):

        # can't generate text without first fitting model
        model = Chlengmo(n=3)
        with self.assertRaises(Exception) as context:
            model.generate(length=99)
        assert isinstance(context.exception, ModelNotFittedError)

    def test_ladder(self):

        # retrieve corpus from NLTK
        filename = "melville-moby_dick.txt"
        text = gutenberg.raw(filename)
        start = "Call me Ishmael"
        start_idx = text.index(start)
        text = text[start_idx:]

        # UTIL: create, fit, and use model
        def bootstrap(n, length, prompt):
            model = Chlengmo(n=n)
            model.fit(text)
            fake_text = model.generate(length=length, prompt=prompt)
            return fake_text

        # generate fake text from incrementing n-gram models
        # NOTE: covers special cases n=0, n=1
        for n in range(25):
            _ = bootstrap(n, length=999, prompt="Call me ")
