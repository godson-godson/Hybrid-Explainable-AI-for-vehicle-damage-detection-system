import os
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"

import sys
from PIL import Image, ImageDraw

def make_sample_doc(path: str):
    img = Image.new("RGB", (800, 500), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((50, 40), "FORM 23 - CERTIFICATE OF REGISTRATION", fill=(0, 0, 0))
    draw.text((50, 90), "REGISTRATION NUMBER: DL 03 CC 4921", fill=(0, 0, 0))
    draw.text((50, 140), "OWNER NAME: RAJESH KUMAR SHARMA", fill=(0, 0, 0))
    draw.text((50, 190), "CHASSIS NUMBER: MALBB51BLAM123456", fill=(0, 0, 0))
    draw.text((50, 240), "ENGINE NUMBER: G4LAJ123456", fill=(0, 0, 0))
    draw.text((50, 290), "REGISTRATION DATE: 14/08/2021", fill=(0, 0, 0))
    draw.text((50, 340), "FUEL: PETROL", fill=(0, 0, 0))
    draw.text((50, 390), "CLASS: LMV-MOTOR CAR", fill=(0, 0, 0))
    img.save(path, format="JPEG")
    print(f"Sample doc saved to {path}")

def main():
    doc_path = "/tmp/sample_rc_test.jpg"
    make_sample_doc(doc_path)

    print("\n--- Initializing PaddleOCR ---")
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang='en', enable_mkldnn=False)
    print("Initialized PaddleOCR successfully.")

    print("\n--- Running ocr.ocr() / predict ---")
    res = ocr.ocr(doc_path)
    print("Type of result:", type(res))
    print("Length of result:", len(res) if res is not None else "None")
    import pprint
    pprint.pprint(res)

if __name__ == "__main__":
    main()
