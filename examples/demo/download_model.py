"""Download the released model and verify its manifest without GPU inference."""
import argparse
from huggingface_hub import snapshot_download
from hs_guard.runtime import verify_model

parser = argparse.ArgumentParser()
parser.add_argument('--output', default='./models/HS-Guard-v10')
args = parser.parse_args()
path = snapshot_download('KrisLiu16/HS-Guard-v10',
    revision='8f7277d6e2900a21bec8e7128ae5ae009469a173', local_dir=args.output,
    allow_patterns=['hs_guard.json', 'model.safetensors', 'config.json', 'tokenizer.json',
                    'tokenizer_config.json', 'vocab.json', 'merges.txt', 'LICENSE', 'BASE_MODEL_LICENSE', 'NOTICE'])
meta = verify_model(path)
print(meta['model_name'], meta['files']['model.safetensors'])
