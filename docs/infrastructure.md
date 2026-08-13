# OpenNICF infrastructure inventory

This document is the living inventory for OpenNICF infrastructure that is
relevant to the current project/stack. It records the verified Proxmox/LXC
surface, the intended role of each asset, and the evidence source used to
confirm it.

Update rule:
- Append new infrastructure here when it is created or verified.
- Do not rewrite historical entries unless the source evidence changes.
- Keep production and evaluation assets separated in the record.

## Verified control surface

| Asset | Role | Status | Evidence |
| --- | --- | --- | --- |
| `hyper` | Proxmox control surface only | verified by workspace boundary notes | `/home/claw/.openclaw/workspace-opennicf/TOOLS.md` |

## OpenNICF LXC inventory

| LXC | Hostname / role | Status | Notes | Evidence |
| --- | --- | --- | --- | --- |
| `156` | PostgreSQL / pgvector | owned, stateful, safety gates apply | Single authoritative database host for OpenNICF knowledge storage | `docs/adr/0004-knowledge-platform.md` |
| `301` | model router / QwenAgent runtime | owned | application runtime lane | workspace boundary notes |
| `302` | application, knowledge, ingestion, persistent mounts | owned | primary app/knowledge runtime lane | workspace boundary notes |
| `305` | production embedding worker | owned, production safety gates apply | production embedding worker lane | workspace boundary notes |
| `308` | `opennicf-embedding-eval` | owned, isolated, `onboot=0` | evaluation guest; no production traffic | `evaluation/issue101-execution-plan.json`, `docs/issue86-cuda-execution-2026-08-12.json` |

## Supporting but not owned

| LXC | Role | Status | Notes |
| --- | --- | --- | --- |
| `151` | supporting | not owned | may be used only when an OpenNICF issue explicitly requires the existing service path |
| `153` | supporting | not owned | may be used only when an OpenNICF issue explicitly requires the existing service path |
| `155` | supporting | not owned | may be used only when an OpenNICF issue explicitly requires the existing service path |

## Excluded

| Asset | Status | Notes |
| --- | --- | --- |
| `154` and unrelated guests | excluded from OpenNICF scope | do not use as access bridges or dependencies |

## Evaluation host evidence

The last recorded CUDA evaluation evidence for LXC 308 is captured in
`docs/issue86-cuda-execution-2026-08-12.json` and records:

- NVIDIA GeForce GTX 1060 3GB
- GPU UUID `GPU-cf150279-b259-0608-019f-8f51e0664ff4`
- driver `535.261.03`
- driver / CUDA compatibility `12.2`
- PyTorch CUDA runtime `12.1`
- container GPU visibility enabled
- minimal tensor operation passed

The same evidence file also records the evaluation isolation rule:

- Proxmox LXC 308 `opennicf-embedding-eval`
- no production traffic
- `onboot=0`

## Change log

- 2026-08-13: created the initial infra inventory from verified repository
  evidence.
