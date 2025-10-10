# test_qwen_export.py
import pandas as pd
import json
import sys
sys.path.insert(0, '.')

from csv_to_json_dataset import run_qwen_mode, build_argparser

# Create sample CSV with your data structure
sample_data = [{
    'sku_id': 3,
    'image_path': '3_836ad79412964175b2df3afaa3e3d59c.png',
    'caption_text': 'Đây là một gói bánh ChocoPN phủ sô cô la. Bánh có lớp kem ở giữa.',
    'keywords': "['bánh', 'choco pn', 'sô cô la', 'bánh quy']",
    'colors': "['đỏ', 'nâu', 'trắng', 'vàng']",
    'shapes': "['tròn']",
    'materials': "['sô cô la', 'bột mì', 'kem']",
    'packaging': "['gói']",
    'taste': "['sô cô la', 'ngọt']",
    'texture': "['mềm', 'xốp']",
    'brand_guess': 'chocopn',
    'variant_guess': 'bánh phủ sô cô la',
    'size_guess': None,
    'category_guess': 'bánh quy',
    'facet_scores': "[{'facet': 'deliciousness', 'score': 0.9}, {'facet': 'convenience', 'score': 0.7}]"
}]

# Write to CSV
df = pd.DataFrame(sample_data)
df.to_csv('test_sample.csv', sep=';', index=False)

# Mock args
class MockArgs:
    def __init__(self):
        self.mode = 'qwen'
        self.image_col = 'image_path'
        self.caption_col = 'caption_text'
        self.base_image_dir = 'uploads'
        self.download_from_url = False
        self.require_caption = True
        self.require_image = True
        self.system_prompt = 'Bạn là AI trích xuất facet sản phẩm từ ảnh và văn bản tiếng Việt.'
        self.json_out = None
        self.jsonl_out = 'test_output.jsonl'

# Run export
args = MockArgs()
run_qwen_mode(df, args)

# Check output
with open('test_output.jsonl', 'r', encoding='utf-8') as f:
    result = json.loads(f.readline())
    
print("Generated Qwen record:")
print(json.dumps(result, ensure_ascii=False, indent=2))