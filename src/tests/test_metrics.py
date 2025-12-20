import unittest
from src.metrics.utils import calc_cer, calc_wer

class TestMetrics(unittest.TestCase):
    def test_cer_basic(self):
        target = "hello"
        pred = "hello"
        self.assertAlmostEqual(calc_cer(target, pred), 0.0)
        
        target = "hello"
        pred = "helo"

        self.assertAlmostEqual(calc_cer(target, pred), 0.2)
        
        target = "hello"
        pred = "pello"

        self.assertAlmostEqual(calc_cer(target, pred), 0.2)

    def test_cer_empty(self):
        self.assertAlmostEqual(calc_cer("", ""), 0.0)
        self.assertAlmostEqual(calc_cer("", "abc"), 1.0)
        self.assertAlmostEqual(calc_cer("abc", ""), 1.0)

    def test_wer_basic(self):
        target = "i have a dog"
        pred = "i have a dog"
        self.assertAlmostEqual(calc_wer(target, pred), 0.0)
        
        target = "i have a dog"
        pred = "i have a cat"
        self.assertAlmostEqual(calc_wer(target, pred), 0.25)
        
        target = "i have a dog"
        pred = "i dog"
        self.assertAlmostEqual(calc_wer(target, pred), 0.5)

    def test_wer_empty(self):
        self.assertAlmostEqual(calc_wer("", ""), 0.0)
        self.assertAlmostEqual(calc_wer("", "hello world"), 1.0)
        self.assertAlmostEqual(calc_wer("hello world", ""), 1.0)

    def test_wer_extra_words(self):
        target = "hello"
        pred = "hello world"
        self.assertAlmostEqual(calc_wer(target, pred), 1.0)

if __name__ == "__main__":
    unittest.main()

