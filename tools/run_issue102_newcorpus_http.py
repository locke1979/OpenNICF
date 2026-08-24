#!/usr/bin/env python3
"""Serial, checkpointed, fail-closed HTTP embedding runner for corpus v1."""
from __future__ import annotations
import argparse, hashlib, json, math, os, pathlib, tempfile, urllib.request

MODEL = "text-embedding-qwen3-embedding-4b"
NATIVE = 2560
EVAL = 768

def digest_bytes(data: bytes) -> str: return hashlib.sha256(data).hexdigest()
def digest(value: object) -> str: return digest_bytes(json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode())
def post(url: str, text: str, timeout: float) -> dict:
    payload = json.dumps({"model": MODEL, "input": [text]}, ensure_ascii=False).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/embeddings", data=payload, headers={"Accept":"application/json","Content-Type":"application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = response.read()
    result = json.loads(body.decode("utf-8"))
    if not isinstance(result, dict) or result.get("model") != MODEL or result.get("fallback"):
        raise RuntimeError("model identity/fallback violation")
    data = result.get("data")
    if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict) or data[0].get("index") not in (None, 0):
        raise RuntimeError("malformed or misaligned response")
    vector = data[0].get("embedding")
    if not isinstance(vector, list) or len(vector) != NATIVE:
        raise RuntimeError(f"native dimension violation: {len(vector) if isinstance(vector, list) else 'invalid'}")
    vector = [float(x) for x in vector]
    if not all(math.isfinite(x) for x in vector) or not any(x != 0.0 for x in vector):
        raise RuntimeError("nonfinite or zero vector")
    prefix = vector[:EVAL]
    norm = math.sqrt(sum(x*x for x in prefix))
    if not math.isfinite(norm) or norm == 0.0: raise RuntimeError("zero evaluation prefix")
    return {"native":vector,"derived":[x/norm for x in prefix],"response_model":result.get("model"),"native_sha256":digest(vector)}
def load_jsonl(path: pathlib.Path, text_key: str, id_key: str) -> list[dict]:
    rows=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(not isinstance(x.get(id_key), str) or not isinstance(x.get(text_key), str) for x in rows): raise RuntimeError(f"invalid input {path}")
    return rows
def atomic(path: pathlib.Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, separators=(",", ":")); f.write("\n"); tmp=pathlib.Path(f.name)
    os.replace(tmp, path)
def run_group(args: argparse.Namespace, name: str, rows: list[dict], id_key: str, text_key: str) -> dict:
    checkpoint=args.checkpoint / f"{name}.json"
    state=json.loads(checkpoint.read_text()) if checkpoint.exists() else {"group":name,"model":MODEL,"native_dimension":NATIVE,"evaluation_dimension":EVAL,"embedding_space_id":"issue102-qwen3-4b-http-2560-v1","data":[]}
    accepted={x["id"]:x for x in state.get("data",[])}
    for index,row in enumerate(rows):
        ident=row[id_key]
        if ident in accepted: continue
        result=post(args.endpoint,row[text_key],args.timeout)
        accepted[ident]={"id":ident,"sequence_index":index,"input_sha256":digest_bytes(row[text_key].encode()),"native_vector_sha256":result["native_sha256"],"native_dimension":NATIVE,"native_embedding":result["native"],"dimension":EVAL,"embedding":result["derived"],"embedding_space_id":"issue102-qwen3-4b-http-2560-v1","response_model":result["response_model"]}
        state["data"]=[accepted[x[id_key]] for x in rows if x[id_key] in accepted]
        atomic(checkpoint,state)
    state["data"]=[accepted[x[id_key]] for x in rows]
    state["count"]=len(state["data"]); state["complete"]=state["count"]==len(rows); state["input_count"]=len(rows)
    atomic(checkpoint,state)
    return state
def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--endpoint",required=True); ap.add_argument("--input-root",type=pathlib.Path,required=True); ap.add_argument("--checkpoint",type=pathlib.Path,required=True); ap.add_argument("--timeout",type=float,default=30.0); args=ap.parse_args()
    docs=load_jsonl(args.input_root/"documents.jsonl","text","document_id"); queries=load_jsonl(args.input_root/"queries.jsonl","query_text","query_id")
    if len(docs)!=326 or len(queries)!=190: raise SystemExit("input count mismatch")
    for name,rows,ik,tk in (("documents",docs,"document_id","text"),("queries",queries,"query_id","query_text")):
        state=run_group(args,name,rows,ik,tk); print(json.dumps({"group":name,"count":state["count"],"complete":state["complete"]}),flush=True)
if __name__=="__main__": main()
