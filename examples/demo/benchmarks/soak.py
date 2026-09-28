"""Replay benign text through HTTP while rotating session IDs."""
import argparse
import concurrent.futures
import json
import os
import threading
import time
from pathlib import Path
from urllib.request import Request, urlopen

parser = argparse.ArgumentParser()
parser.add_argument('--url', default='http://127.0.0.1:8080')
parser.add_argument('--seconds', type=int, default=300)
parser.add_argument('--workers', type=int, default=16)
parser.add_argument('--output', required=True)
args = parser.parse_args()
headers={'Content-Type':'application/json'}
if os.environ.get('GUARD_ACCESS_TOKEN'):
    headers['Authorization']='Bearer '+os.environ['GUARD_ACCESS_TOKEN']

def get_info():
    return json.load(urlopen(Request(args.url+'/api/info', headers=headers), timeout=10))

info=get_info()
rules={role:info['presets'][role][0] for role in ('user','assistant')}
end=time.monotonic()+args.seconds
lock=threading.Lock()
counts={'completed':0,'errors':0,'http_errors':0}
latencies=[]
errors=[]
samples=[]

def worker(slot):
    index=0
    while time.monotonic()<end:
        # Unique IDs force eviction beyond the 64-session cache limit.
        session_id=f'soak-{slot}-{index}'
        repetitions=(4,32,128)[(slot+index)%3]
        history=[{'role':'user','content':'Explain photosynthesis and how plants store energy. '*repetitions}]
        body={'session_id':session_id,'messages':history,'mode':'replay',
              'guard':{'action':'observe','user_rule':rules['user'],'assistant_rule':rules['assistant']},
              'replay':{'content':'Plants use sunlight to turn water and carbon dioxide into sugars and oxygen.','chars':3,'interval_ms':2}}
        started=time.monotonic()
        try:
            req=Request(args.url+'/api/chat',data=json.dumps(body).encode(),headers=headers)
            with urlopen(req,timeout=90) as response:
                text=response.read().decode()
            if 'event: error\n' in text or 'event: done\n' not in text:
                raise RuntimeError('incomplete or failed event stream')
            with lock:
                counts['completed']+=1
                latencies.append((time.monotonic()-started)*1000)
        except Exception as error:
            with lock:
                counts['errors']+=1
                counts['http_errors']+=int(hasattr(error,'code'))
                if len(errors)<10:errors.append(type(error).__name__)
        index+=1

started=time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
    futures=[pool.submit(worker,i) for i in range(args.workers)]
    while any(not f.done() for f in futures):
        data=get_info()
        samples.append({'seconds':round(time.monotonic()-started,2),
                        'sessions':data['sessions'],'active':data.get('active_chats'),
                        'latency':data['latency']})
        time.sleep(10)
    for f in futures:f.result()
ordered=sorted(latencies)
report={'seconds':round(time.monotonic()-started,2),'workers':args.workers,**counts,'errors_sample':errors,
        'response_p50_ms':ordered[len(ordered)//2] if ordered else None,
        'response_p95_ms':ordered[min(len(ordered)-1,int(.95*len(ordered)))] if ordered else None,
        'max_sessions':max(x['sessions'] for x in samples),'samples':samples}
Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='samples'},indent=2))
assert report['errors']==0 and report['max_sessions']<=64
