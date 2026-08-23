# CT 305 corpus-recovery audit

Status: `CORPUS_IN_CT305=NOT_FOUND`; `CORPUS_RECOVERY=BLOCKED`

This was a read-only inspection of the authorized Proxmox host `hyper`
(`192.168.1.116`). CT 305 was stopped and was not started. No CT lifecycle,
configuration, service, model, index, database, mount, permission, or
production-data mutation occurred. CT 308 was not touched.

## Container and storage facts

- Proxmox identity: `hyper`, Proxmox VE `8.4.11`, kernel `6.8.12-41-pve`.
- CT 305: `stopped`.
- Host-side rootfs path: `/ssdpool/subvol-305-disk-0`.
- CT rootfs config: `ssdpool:subvol-305-disk-0`, size 16G.
- No `mp*` bind-mount entries were present in `pct config 305`.
- Host storage `ssdpool`: active ZFS pool; no CT 305 ZFS snapshot was listed.
- Project backup storage: `/mnt/raid0store/backups/local`.

## Paths inspected

The rootfs was inspected directly at its existing host-side ZFS path; it was
not mounted or attached. Bounded metadata checks covered:

- `/var/lib/opennicf`, including `eval`, `corpus`, `embeddings`, `vectors`,
  `manifests`, `checkpoints`, and `results`;
- `/opt/opennicf`;
- `/etc/systemd/system`, `/etc/opennicf`, cron directories, `/mnt`, `/srv`, and
  `/data`;
- project-owned backup metadata under `/ssdpool/backups` and
  `/mnt/raid0store/backups/local`.

The only relevant live data found was a Qwen 0.6B model cache under
`/var/lib/opennicf/models`. No corpus, query, manifest, label, checkpoint, or
vector filenames matching the requested recovery terms were present.

The embedding service unit is
`/etc/systemd/system/opennicf-embedding-worker.service`; it references
`/var/lib/opennicf` as its writable path and `/etc/opennicf/embedding-worker.env`
as an environment file. Only environment variable names were inspected; values
and service tokens were not printed.

## Candidate backup

One CT 305 backup is present:

- path: `/mnt/raid0store/backups/local/dump/vzdump-lxc-305-2026_08_22-20_31_25.tar.zst`;
- type: Zstandard-compressed vzdump archive;
- size: `5,850,507,957` bytes;
- modification time: `2026-08-22 20:36:27 UTC`;
- sidecars: matching `.log` and `.notes` files;
- SHA-256: not obtained in this bounded session (the read-only hash command
  returned no digest before the remote session ended);
- corpus validation: not performed; the archive was not restored, mounted,
  extracted, or copied.

The archive is therefore a recovery candidate, not evidence that the
authoritative corpus exists. Its contents require a separately authorized
controlled export or offline inspection. No other CT 305 backup was found in
the inspected backup paths.

## Candidate disposition

No valid artifact proved all of the following: 715 ordered document IDs, 190
ordered query IDs, source text, labels/references, preprocessing identity, and
compatibility with authoritative digest prefix `cb6f30f5`. Model weights,
service configuration, production indexes, and unvalidated backup archives are
not corpus substitutes.

```text
CORPUS_IN_CT305 = NOT_FOUND
CORPUS_RECOVERY = BLOCKED
CT305_TOUCHED = false
PRODUCTION = UNCHANGED
```

Exact next operator action: obtain separate authorization for controlled,
non-destructive inspection/export of the identified CT 305 backup, or restore
the original sanitized corpus from its owner. Do not restore or mount the
backup during this audit.
