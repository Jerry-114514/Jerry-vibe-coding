import base64
import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import run_teacher_role_calibration as calibration


class ImageBudgetTests(unittest.TestCase):
    def encode(self,image):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'data').mkdir()
            image.save(root/'data/image.png')
            item={'image_paths':['data/image.png'],'question':'Locate the red object'}
            with patch.object(calibration,'ROOT',root):
                blocks,hashes=calibration.content(item)
            raw=base64.b64decode(blocks[1]['image_url']['url'].split(',',1)[1])
            with Image.open(io.BytesIO(raw)) as output:
                size=output.size
            return raw,size,hashes[0]

    def test_small_image_preserves_original_png_bytes(self):
        image=Image.new('RGB',(128,128),'red')
        expected=io.BytesIO();image.save(expected,format='PNG')
        raw,size,metadata=self.encode(image)
        self.assertEqual(raw,expected.getvalue())
        self.assertEqual(size,(128,128))
        self.assertEqual(metadata['transmitted_png_sha256'],hashlib.sha256(raw).hexdigest())

    def test_pixel_limit_is_applied_before_sending(self):
        raw,size,metadata=self.encode(Image.new('RGB',(2560,1708),'red'))
        self.assertLessEqual(size[0]*size[1],4000000)
        self.assertEqual(metadata['source_size'],[2560,1708])
        self.assertAlmostEqual(size[0]/size[1],2560/1708,places=2)

    def test_encoded_byte_limit_is_applied_even_below_pixel_limit(self):
        pixels=np.random.default_rng(42).integers(0,256,size=(1900,2000,3),dtype=np.uint8)
        raw,size,metadata=self.encode(Image.fromarray(pixels))
        self.assertLessEqual(len(raw),8000000)
        self.assertLess(size[0]*size[1],1900*2000)
        self.assertEqual(metadata['source_size'],[2000,1900])


if __name__=='__main__':
    unittest.main()
