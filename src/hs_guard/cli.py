import argparse
import json
from .runtime import MODEL_ID,load_model,score_messages,verify_model

def main():
    p=argparse.ArgumentParser(description='Score a message with HS-Guard on a CUDA GPU.')
    p.add_argument('--model',default=MODEL_ID)
    p.add_argument('--head',choices=['redline','general'],default='redline')
    p.add_argument('--text')
    p.add_argument('--verify-only',action='store_true')
    args=p.parse_args()
    if args.verify_only:
        m=verify_model(args.model);print(json.dumps({'verified':True,'model':m['model_name']}));return
    if not args.text:p.error('--text is required unless --verify-only is used')
    model,tok,meta=load_model(args.model,head=args.head)
    print(json.dumps(score_messages(model,tok,meta,[{'role':'user','content':args.text}])))

if __name__=='__main__':main()
