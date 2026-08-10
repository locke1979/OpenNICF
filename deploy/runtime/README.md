# OpenNICF runtime release framework

This framework packages a versioned source artifact, verifies a SHA-256
manifest, installs it without overwriting an existing version, switches the
`/opt/opennicf/current` symlink atomically, restarts one systemd service, and
requires a bounded readiness URL before accepting the release. A failed gate
restores the previous symlink and attempts to restart the previous service.

Runtime state is kept outside the release archive:

- `/opt/opennicf/releases/<version>/`
- `/opt/opennicf/current`
- `/etc/opennicf/*.env` (mode 0600, never packaged)
- `/var/lib/opennicf/`
- `/var/log/opennicf/`

`build-release` is deterministic and emits an artifact hash. `install-release`
refuses to overwrite a version and verifies the manifest before activation.
`opennicf.release` exposes the verify/activate/rollback state machine for
controlled runners and is testable with `OPENNICF_SYSTEMCTL`.

After activation, `install-units` installs the service templates and reloads
systemd. `rollback SERVICE HEALTH_URL` invokes the same readiness-gated
rollback path. The installer creates the locked `opennicf` service account and
owns only the state/log directories; protected environment files remain an
operator-provided step.

The installer retains `/opt`, `/etc`, and `/var` as production defaults, while
`OPENNICF_RUNTIME_ROOT`, `OPENNICF_ETC_ROOT`, `OPENNICF_STATE_ROOT`, and
`OPENNICF_LOG_ROOT` allow an isolated runner to validate installation without
root access.

The service templates intentionally reference `opennicf.service_runtime`, a
bounded health/runtime entrypoint. Database, object-store, and embedding
configuration is supplied only through protected environment files.
