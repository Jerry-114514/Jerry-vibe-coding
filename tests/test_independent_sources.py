import io
from pathlib import Path
import sys
import unittest

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from index_independent_sources import image_facts, normalized_options


class IndependentSourceTests(unittest.TestCase):
    def test_six_choices_are_preserved(self):
        values = ['one', 'two', 'three', 'four', 'five', 'six']
        self.assertEqual(normalized_options(repr(values)), values)

    def test_option_strings_are_data_not_executable_code(self):
        with self.assertRaises((ValueError, SyntaxError)):
            normalized_options('__import__("os").getcwd()')

    def test_rgb_duplicate_survives_lossless_reencoding(self):
        image = Image.new('RGB', (13, 7), (25, 50, 200))
        first, second = io.BytesIO(), io.BytesIO()
        image.save(first, format='PNG', compress_level=0)
        image.save(second, format='PNG', compress_level=9)
        a, b = image_facts(first.getvalue()), image_facts(second.getvalue())
        self.assertNotEqual(a['source_sha256'], b['source_sha256'])
        self.assertEqual(a['pixel_sha256'], b['pixel_sha256'])
        self.assertEqual((a['width'], a['height']), (13, 7))


if __name__ == '__main__':
    unittest.main()
