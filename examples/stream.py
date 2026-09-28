from hs_guard import StreamingGuard,encode_messages,load_model

model,tokenizer,metadata=load_model()
messages=[{'role':'user','content':'How can I improve my study habits?'},
          {'role':'assistant','content':'Set a regular schedule and take short breaks.'}]
ids,positions,role=encode_messages(tokenizer,messages)
stream=StreamingGuard(model,metadata,max_slots=1)
start=positions[0]
if start:stream.step([(0,ids[:start])],role=role)
for offset in range(start,len(ids),16):
    print(stream.step([(0,ids[offset:offset+16])],role=role))
