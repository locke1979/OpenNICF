#!/usr/bin/env python3
"""Evaluate CUDA-generated dense and lexical/RRF Issue #86 text arms."""
from __future__ import annotations
from collections import defaultdict
import json, math, re, sys
from pathlib import Path

from opennicf.experimental_reporting import aggregate_metrics, paired_bootstrap_interval, per_query_retrieval_metrics


def tokens(text): return re.findall(r"[a-z0-9_./:-]+", text.casefold())


def rrf(dense, lexical, k=60):
    scores=defaultdict(float)
    for ranking in (dense,lexical):
        for rank,item in enumerate(ranking,1): scores[item]+=1/(k+rank)
    return sorted(scores,key=lambda item:(-scores[item],item))


def main():
    corpus=json.loads(Path(sys.argv[1]).read_text()); labels=json.loads(Path(sys.argv[2]).read_text())["labels"]
    documents={item["chunk_id"]:item for item in corpus["chunks"]}
    strata=defaultdict(list)
    for item in labels:
        qid=item["query_id"]; strata[item["leakage_stratum"]].append(qid)
        for category in item.get("categories",[]): strata[category].append(qid)
        if item.get("hard_negative") and qid not in strata["HARD_NEGATIVE"]: strata["HARD_NEGATIVE"].append(qid)
    report={"evaluation_mode":"AUTOMATED_NON_HUMAN_REVIEWED","model_execution_device":"CUDA_REQUIRED","arms":{}}
    per_arm={}
    for arm,path,hybrid in (("T00",sys.argv[3],False),("T01",sys.argv[4],False),("T04",sys.argv[3],True),("T05",sys.argv[4],True)):
        vectors=json.loads(Path(path).read_text()); docs={x["id"]:x["vector"] for x in vectors["documents"]}; queries={x["id"]:x["vector"] for x in vectors["queries"]}
        per={}; rankings={}
        for label in labels:
            qid=label["query_id"]; q=queries[qid]
            dense=sorted(docs,key=lambda item:(-sum(a*b for a,b in zip(q,docs[item])),item))
            qt=set(tokens(label["automated_query_text"])); lexical=sorted(documents,key=lambda item:(-len(qt & set(tokens(documents[item]["text"]))),item))
            ranking=rrf(dense[:50],lexical[:50]) if hybrid else dense
            relevant=label["final_relevant_ids"]; negative=[label["hard_negative"]["chunk_id"]] if label.get("hard_negative") else []
            per[qid]=per_query_retrieval_metrics(ranking,relevant,hard_negative_ids=negative)
            rankings[qid]=ranking[:50]
        per_arm[arm]=per
        report["arms"][arm]={"status":"EXECUTED_CUDA","provenance":vectors["provenance"],"metrics":aggregate_metrics(per,strata),"rankings":rankings}
    report["paired_bootstrap"]={}
    for baseline,candidate in (("T00","T01"),("T00","T04"),("T01","T05")):
        report["paired_bootstrap"][f"{candidate}_minus_{baseline}"]={metric:paired_bootstrap_interval({q:v[metric] for q,v in per_arm[baseline].items()},{q:v[metric] for q,v in per_arm[candidate].items()}) for metric in ("recall_at_5","mrr_at_10","ndcg_at_10","hard_negative_fp_at_5")}
    report["warnings"]=["AUTOMATED LABELS MAY CONTAIN SOURCE-DERIVED LEAKAGE","QUALITY RESULTS ARE COMPARATIVE EXPERIMENTAL EVIDENCE","PRODUCTION REMAINS UNCHANGED"]
    report["winner_selection"]="BLOCKED"; report["production"]="UNCHANGED"; report["deployments"]=0; report["merges"]=0
    Path(sys.argv[5]).write_text(json.dumps(report,separators=(",",":")))

if __name__=="__main__": main()
