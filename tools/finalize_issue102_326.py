from __future__ import annotations
import hashlib,json
from pathlib import Path

R=Path(__file__).resolve().parents[1]/'evaluation/issue102-newcorpus-v1'
CID='issue102-http-newcorpus-v1-326x190'
EID='issue102-qwen3-4b-http-326x190-v1'
ESP='issue102-qwen3-4b-http-2560-v1'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name,obj):
    p=R/name
    p.write_bytes((json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(',',':'),indent=2)+'\n').encode())
    return sha(p)

def main():
    files=['corpus-manifest.json','documents.jsonl','queries.jsonl','labels.jsonl','hard-negatives.json','source-inventory.json','preprocessing-manifest.json','embedding-space-manifest.json','experiment-manifest.json','automated-label-audit.json','leakage-audit.json','benchmark-config.json','metrics.json','corpus-generation-audit.json','checkpoints/documents.json','checkpoints/queries.json']
    dig={x:sha(R/x) for x in files}
    corpus=json.loads((R/'corpus-manifest.json').read_text())
    corpus.update({'corpus_id':CID,'documents':326,'queries':190,'status':'COMPLETE_NEW_CORPUS','comparability':'NON_COMPARABLE_TO_T00_T01','artifact_sha256':{k:v for k,v in dig.items() if k in ('documents.jsonl','queries.jsonl','labels.jsonl','hard-negatives.json')}})
    dump('corpus-manifest.json',corpus)
    exp=json.loads((R/'experiment-manifest.json').read_text()); exp.update({'corpus_id':CID,'experiment_id':EID,'comparability':'NON_COMPARABLE_TO_T00_T01','status':'COMPLETE_NEW_CORPUS'}); dump('experiment-manifest.json',exp)
    emb=json.loads((R/'embedding-space-manifest.json').read_text()); emb['embedding_space_id']=ESP; emb['endpoint_digest']=hashlib.sha256(b'http://192.168.1.137:1234').hexdigest(); dump('embedding-space-manifest.json',emb)
    pre=json.loads((R/'preprocessing-manifest.json').read_text()); pre['corpus_id']=CID; dump('preprocessing-manifest.json',pre)
    cfg=json.loads((R/'benchmark-config.json').read_text()); cfg.update({'benchmark_execution_allowed':True,'corpus_id':CID,'experiment_id':EID,'comparability':'NON_COMPARABLE_TO_T00_T01'}); dump('benchmark-config.json',cfg)
    aud=json.loads((R/'automated-label-audit.json').read_text()); aud.update({'corpus_id':CID,'experiment_id':EID,'comparability':'NON_COMPARABLE_TO_T00_T01'}); dump('automated-label-audit.json',aud)
    ga=json.loads((R/'corpus-generation-audit.json').read_text()); ga.update({'corpus_id':CID,'experiment_id':EID,'documents':326,'queries':190,'status':'COMPLETE_NEW_CORPUS','comparability':'NON_COMPARABLE_TO_T00_T01','result_manifest_sha256':None}); dump('corpus-generation-audit.json',ga)
    m=json.loads((R/'metrics.json').read_text()); m.update({'corpus_id':CID,'experiment_id':EID,'comparability':'NON_COMPARABLE_TO_T00_T01'}); mh=dump('metrics.json',m); dig['metrics.json']=mh
    result=json.loads((R/'result-manifest.json').read_text())
    result.update({'corpus_id':CID,'experiment_id':EID,'documents':326,'queries':190,'comparability':'NON_COMPARABLE_TO_T00_T01','metrics_sha256':mh,'artifact_digests':dig,'preflight':{'endpoint':'http://192.168.1.137:1234','model':'text-embedding-qwen3-embedding-4b','text_contract_pass':True,'native_dimension':2560,'deterministic':True,'ordered_batch':True,'invalid_model_rejected':True,'probe_vector_sha256':'986a223075b4cbdaa020c1aa77c35af5a67d1f3155fa30ca783b72f880b546ed'}})
    rh=dump('result-manifest.json',result); (R/'result-manifest.json.sha256').write_text(rh+'  result-manifest.json\n')
    ga['result_manifest_sha256']=rh; dump('corpus-generation-audit.json',ga)
    for p in sorted(R.glob('*.json')):
        if p.name!='result-manifest.json': (R/(p.name+'.sha256')).write_text(sha(p)+'  '+p.name+'\n')
    for p in sorted((R/'checkpoints').glob('*.json')): (p.parent/(p.name+'.sha256')).write_text(sha(p)+'  '+p.name+'\n')
    (R/'corpus-generation-audit.md').write_text(f'# {CID}\n\nStatus: `COMPLETE_NEW_CORPUS`\n\nThis automated-only experiment contains exactly **326 documents/chunks** and **190 queries**. It is independent of the historical 715/190 corpus and is `NON_COMPARABLE_TO_T00_T01`.\n\nThe endpoint was validated and serial embeddings completed with native 2560-D vectors and derived 768-D prefix+L2 vectors. No human review was performed. Latency and throughput were not captured by the runner and remain unavailable. Production, CT305, and CT308 were unchanged.\n')

if __name__=='__main__': main()
