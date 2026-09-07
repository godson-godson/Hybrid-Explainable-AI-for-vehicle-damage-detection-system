import os
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"

import time
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_textline_orientation=True, lang='en', enable_mkldnn=False)

# Warmup run
_ = ocr.ocr("/tmp/sample_rc_test.jpg")

# Timed run
t0 = time.perf_counter()
res = ocr.ocr("/tmp/sample_rc_test.jpg")
t1 = time.perf_counter()

print(f"Warm inference latency: {(t1 - t0)*1000:.2f} ms")
