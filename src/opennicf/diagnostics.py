"""Secure pull-based on-prem diagnostics broker and worker helpers."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import csv
import hmac
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence
import uuid

from .knowledge import KnowledgePlatform

APPROVED_WRAPPER = "approved-sqlcl-wrapper.ps1"
ALLOWED_OPERATION_CLASSES = {"read_select", "describe"}
_DANGEROUS_SQL = re.compile(
    r"\b("
    r"insert|update|delete|merge|create|alter|drop|truncate|grant|revoke|commit|rollback|execute|exec|call|declare|begin|end|"
    r"shutdown|replace|attach|detach|vacuum|analyze"
    r")\b",
    re.I,
)
_MULTI_STATEMENT = re.compile(r";\s*\S")
_PLACEHOLDER = re.compile(r"{{\s*([A-Za-z_][A-Za-z0-9_]*)\s*}}")


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _json_default(value: Any):
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dict__"):
        return asdict(value)
    return str(value)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default)


def _sha256_text(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    raise TypeError(f"unsupported SQL literal value: {type(value).__name__}")


def validate_read_only_sql(sql_text: str) -> str:
    candidate = sql_text.strip()
    if not candidate:
        raise ValueError("sql_text cannot be empty")
    if _MULTI_STATEMENT.search(candidate):
        raise ValueError("multiple SQL statements are not allowed")
    if _DANGEROUS_SQL.search(candidate):
        raise ValueError("write-capable SQL is not allowed")
    normalized = candidate.upper()
    if not (normalized.startswith("SELECT") or normalized.startswith("WITH") or normalized.startswith("DESCRIBE") or normalized.startswith("DESC")):
        raise ValueError("only read-only SELECT/DESCRIBE statements are allowed")
    return candidate


@dataclass(frozen=True)
class DiagnosticTemplate:
    template_id: str
    sql: str
    description: str = ""
    expected_output_format: str = "csv"
    allowed_parameters: tuple[str, ...] = ()

    def render(self, parameters: Mapping[str, Any] | None = None) -> str:
        parameters = dict(parameters or {})
        found = set(_PLACEHOLDER.findall(self.sql))
        allowed = set(self.allowed_parameters or tuple(found))
        if found - allowed:
            raise ValueError(f"template {self.template_id} contains undeclared parameters: {sorted(found - allowed)}")
        missing = allowed - set(parameters)
        if missing:
            raise ValueError(f"template {self.template_id} is missing parameters: {sorted(missing)}")
        unexpected = set(parameters) - allowed
        if unexpected:
            raise ValueError(f"template {self.template_id} received unexpected parameters: {sorted(unexpected)}")

        def replace(match: re.Match[str]) -> str:
            return _sql_literal(parameters[match.group(1)])

        rendered = _PLACEHOLDER.sub(replace, self.sql)
        return validate_read_only_sql(rendered)


@dataclass(frozen=True)
class DiagnosticJob:
    job_id: str
    audit_id: str | None
    finding_id: str | None
    database_profile_alias: str
    operation_class: str
    wrapper_name: str
    sql_text: str
    expected_output_format: str
    max_rows: int
    max_runtime_seconds: float
    max_output_bytes: int
    created_at: datetime
    expires_at: datetime
    template_id: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    payload_hash: str = ""
    signature: str = ""

    def immutable_payload(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "audit_id": self.audit_id,
            "finding_id": self.finding_id,
            "database_profile_alias": self.database_profile_alias,
            "operation_class": self.operation_class,
            "wrapper_name": self.wrapper_name,
            "sql_text": self.sql_text,
            "expected_output_format": self.expected_output_format,
            "max_rows": self.max_rows,
            "max_runtime_seconds": self.max_runtime_seconds,
            "max_output_bytes": self.max_output_bytes,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "template_id": self.template_id,
            "parameters": self.parameters,
            "metadata": self.metadata,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.immutable_payload()
        payload["payload_hash"] = self.payload_hash
        payload["signature"] = self.signature
        return payload


@dataclass(frozen=True)
class DiagnosticLease:
    job: DiagnosticJob
    worker_id: str
    lease_token: str
    lease_expires_at: datetime
    attempt_number: int


@dataclass(frozen=True)
class DiagnosticResult:
    job_id: str
    lease_token: str
    worker_id: str
    stdout: str
    stderr: str
    exit_code: int
    row_count: int
    runtime_seconds: float
    truncated: bool
    stdout_hash: str
    stderr_hash: str
    output_hash: str
    result_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


@dataclass(frozen=True)
class SqlclFixture:
    stdout: str
    stderr: str = ""
    exit_code: int = 0
    runtime_seconds: float = 0.1
    output_format: str = "csv"


class DiagnosticTemplateRegistry:
    def __init__(self, templates: Sequence[DiagnosticTemplate] | None = None):
        self._templates = {template.template_id: template for template in templates or ()}

    def register(self, template: DiagnosticTemplate) -> None:
        self._templates[template.template_id] = template

    def resolve(self, template_id: str) -> DiagnosticTemplate:
        try:
            return self._templates[template_id]
        except KeyError as exc:
            raise KeyError(f"unknown diagnostic template: {template_id}") from exc


class DiagnosticPolicyError(RuntimeError):
    pass


class DiagnosticLeaseError(RuntimeError):
    pass


class DiagnosticReplayError(RuntimeError):
    pass


class DiagnosticBrokerStore:
    def __init__(self, path: str):
        self.path = path
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS diagnostic_jobs (
                    job_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    signature TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    leased_by TEXT,
                    leased_at TEXT,
                    lease_expires_at TEXT,
                    lease_token TEXT,
                    result_json TEXT,
                    result_hash TEXT,
                    stdout_hash TEXT,
                    stderr_hash TEXT,
                    output_hash TEXT,
                    submitted_at TEXT,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS diagnostic_heartbeats (
                    worker_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                """
            )

    def insert_job(self, job: DiagnosticJob, *, status: str = "pending") -> None:
        now = _utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO diagnostic_jobs (
                    job_id, payload_json, payload_hash, signature, status, attempt_count,
                    created_at, expires_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.job_id,
                    canonical_json(job.immutable_payload()),
                    job.payload_hash,
                    job.signature,
                    status,
                    0,
                    job.created_at.isoformat(),
                    job.expires_at.isoformat(),
                    now,
                ),
            )

    def get_job_row(self, job_id: str) -> sqlite3.Row | None:
        with self._connect() as conn:
            return conn.execute("SELECT * FROM diagnostic_jobs WHERE job_id = ?", (job_id,)).fetchone()

    def next_pending_job(self, now: datetime) -> sqlite3.Row | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM diagnostic_jobs
                WHERE status = 'pending' AND expires_at > ?
                ORDER BY created_at ASC
                LIMIT 1
                """,
                (now.isoformat(),),
            ).fetchone()
            if row is None:
                expired = conn.execute(
                    "SELECT job_id FROM diagnostic_jobs WHERE status = 'pending' AND expires_at <= ? ORDER BY created_at ASC LIMIT 1",
                    (now.isoformat(),),
                ).fetchone()
                if expired is not None:
                    conn.execute(
                        "UPDATE diagnostic_jobs SET status = 'expired', updated_at = ? WHERE job_id = ?",
                        (now.isoformat(), expired["job_id"]),
                    )
            return row

    def lease_job(self, job_id: str, worker_id: str, lease_token: str, lease_expires_at: datetime) -> sqlite3.Row:
        now = _utcnow().isoformat()
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM diagnostic_jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            if row["status"] != "pending":
                raise DiagnosticLeaseError("job is not eligible for leasing")
            if datetime.fromisoformat(row["expires_at"]) <= _utcnow():
                conn.execute(
                    "UPDATE diagnostic_jobs SET status = 'expired', updated_at = ? WHERE job_id = ?",
                    (now, job_id),
                )
                raise DiagnosticLeaseError("job has expired")
            attempt_count = int(row["attempt_count"]) + 1
            conn.execute(
                """
                UPDATE diagnostic_jobs
                SET status = 'leased',
                    attempt_count = ?,
                    leased_by = ?,
                    leased_at = ?,
                    lease_expires_at = ?,
                    lease_token = ?,
                    updated_at = ?
                WHERE job_id = ?
                """,
                (attempt_count, worker_id, now, lease_expires_at.isoformat(), lease_token, now, job_id),
            )
            return conn.execute("SELECT * FROM diagnostic_jobs WHERE job_id = ?", (job_id,)).fetchone()

    def complete_job(self, job_id: str, lease_token: str, result: DiagnosticResult) -> sqlite3.Row:
        now = _utcnow().isoformat()
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM diagnostic_jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            if row["status"] == "completed":
                raise DiagnosticReplayError("job result has already been recorded")
            if row["status"] != "leased":
                raise DiagnosticLeaseError("job is not leased")
            if row["lease_token"] != lease_token:
                raise DiagnosticLeaseError("lease token mismatch")
            if row["lease_expires_at"] and datetime.fromisoformat(row["lease_expires_at"]) < _utcnow():
                raise DiagnosticLeaseError("lease expired")
            conn.execute(
                """
                UPDATE diagnostic_jobs
                SET status = 'completed',
                    result_json = ?,
                    result_hash = ?,
                    stdout_hash = ?,
                    stderr_hash = ?,
                    output_hash = ?,
                    submitted_at = ?,
                    updated_at = ?
                WHERE job_id = ?
                """,
                (
                    canonical_json(result.to_dict()),
                    result.result_hash,
                    result.stdout_hash,
                    result.stderr_hash,
                    result.output_hash,
                    now,
                    now,
                    job_id,
                ),
            )
            return conn.execute("SELECT * FROM diagnostic_jobs WHERE job_id = ?", (job_id,)).fetchone()

    def record_heartbeat(self, worker_id: str, status: str, metadata: Mapping[str, Any] | None = None) -> None:
        now = _utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO diagnostic_heartbeats (worker_id, status, observed_at, metadata_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(worker_id) DO UPDATE SET
                    status = excluded.status,
                    observed_at = excluded.observed_at,
                    metadata_json = excluded.metadata_json
                """,
                (worker_id, status, now, canonical_json(dict(metadata or {}))),
            )

    def heartbeat(self, worker_id: str) -> sqlite3.Row | None:
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM diagnostic_heartbeats WHERE worker_id = ?",
                (worker_id,),
            ).fetchone()


class SecureDiagnosticBroker:
    """Signed immutable diagnostic jobs with single-use leases."""

    name = "onprem-diagnostic-broker"

    def __init__(
        self,
        store: DiagnosticBrokerStore,
        *,
        signing_key: bytes,
        templates: DiagnosticTemplateRegistry | None = None,
        lease_seconds: int = 900,
        max_rows: int = 500,
        max_runtime_seconds: float = 30.0,
        max_output_bytes: int = 64_000,
    ):
        self.store = store
        self.signing_key = signing_key
        self.templates = templates or DiagnosticTemplateRegistry()
        self.lease_seconds = lease_seconds
        self.max_rows = max_rows
        self.max_runtime_seconds = max_runtime_seconds
        self.max_output_bytes = max_output_bytes

    def _sign(self, payload: Mapping[str, Any]) -> tuple[str, str]:
        payload_hash = _sha256_text(canonical_json(payload))
        signature = hmac.new(self.signing_key, payload_hash.encode("utf-8"), sha256).hexdigest()
        return payload_hash, signature

    def _validate_request(self, request: Mapping[str, Any]) -> dict[str, Any]:
        operation_class = str(request.get("operation_class") or "").strip()
        if operation_class not in ALLOWED_OPERATION_CLASSES:
            raise DiagnosticPolicyError("diagnostic operation is not allow-listed")
        wrapper_name = str(request.get("wrapper_name") or APPROVED_WRAPPER).strip()
        if wrapper_name != APPROVED_WRAPPER:
            raise DiagnosticPolicyError("only the approved SQLcl wrapper may execute jobs")

        template_id = request.get("template_id")
        sql_text = request.get("sql_text")
        parameters = dict(request.get("parameters") or request.get("template_parameters") or {})
        if template_id:
            try:
                template = self.templates.resolve(str(template_id))
                sql_text = template.render(parameters)
            except (KeyError, ValueError) as exc:
                raise DiagnosticPolicyError(str(exc)) from exc
        elif sql_text:
            try:
                sql_text = validate_read_only_sql(str(sql_text))
            except ValueError as exc:
                raise DiagnosticPolicyError(str(exc)) from exc
        else:
            raise DiagnosticPolicyError("a template_id or sql_text is required")

        max_rows = min(int(request.get("max_rows", self.max_rows)), self.max_rows)
        max_runtime_seconds = min(float(request.get("max_runtime_seconds", self.max_runtime_seconds)), self.max_runtime_seconds)
        max_output_bytes = min(int(request.get("max_output_bytes", self.max_output_bytes)), self.max_output_bytes)
        expires_in_seconds = int(request.get("expires_in_seconds", 300))
        created_at = request.get("created_at")
        created_at = created_at if isinstance(created_at, datetime) else _utcnow()
        expires_at = created_at + timedelta(seconds=max(1, expires_in_seconds))
        payload = {
            "job_id": str(request.get("job_id") or uuid.uuid4().hex),
            "audit_id": request.get("audit_id"),
            "finding_id": request.get("finding_id"),
            "database_profile_alias": str(request.get("database_profile_alias") or request.get("profile_alias") or "default"),
            "operation_class": operation_class,
            "wrapper_name": APPROVED_WRAPPER,
            "sql_text": sql_text,
            "expected_output_format": str(request.get("expected_output_format") or request.get("output_format") or "csv"),
            "max_rows": max_rows,
            "max_runtime_seconds": max_runtime_seconds,
            "max_output_bytes": max_output_bytes,
            "created_at": created_at,
            "expires_at": expires_at,
            "template_id": str(template_id) if template_id else None,
            "parameters": parameters,
            "metadata": dict(request.get("metadata") or {}),
        }
        payload["metadata"].setdefault("request_hash", _sha256_text(canonical_json(dict(request))))
        return payload

    def create_job(self, request: Mapping[str, Any]) -> DiagnosticJob:
        payload = self._validate_request(request)
        payload_hash, signature = self._sign(payload)
        job = DiagnosticJob(**payload, payload_hash=payload_hash, signature=signature)
        self.store.insert_job(job)
        return job

    def request(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        job = self.create_job(payload)
        request_id = _sha256_text(canonical_json(job.immutable_payload()))
        return {
            "status": "pending",
            "job_id": job.job_id,
            "request_id": request_id,
            "broker": self.name,
            "payload": job.to_dict(),
            "signature": job.signature,
        }

    def _job_from_row(self, row: sqlite3.Row) -> DiagnosticJob:
        payload = json.loads(row["payload_json"])
        payload["created_at"] = datetime.fromisoformat(payload["created_at"])
        payload["expires_at"] = datetime.fromisoformat(payload["expires_at"])
        return DiagnosticJob(**payload, payload_hash=row["payload_hash"], signature=row["signature"])

    def verify_job(self, job: DiagnosticJob) -> None:
        payload_hash, signature = self._sign(job.immutable_payload())
        if payload_hash != job.payload_hash or signature != job.signature:
            raise DiagnosticPolicyError("job signature or payload hash mismatch")
        if job.expires_at <= _utcnow():
            raise DiagnosticLeaseError("job has expired")
        try:
            validate_read_only_sql(job.sql_text)
        except ValueError as exc:
            raise DiagnosticPolicyError(str(exc)) from exc
        if job.wrapper_name != APPROVED_WRAPPER:
            raise DiagnosticPolicyError("wrapper is not approved")

    def poll(self, worker_id: str, *, now: datetime | None = None) -> DiagnosticLease | None:
        now = now or _utcnow()
        row = self.store.next_pending_job(now)
        if row is None:
            return None
        job = self._job_from_row(row)
        self.verify_job(job)
        lease_token = uuid.uuid4().hex
        lease_expires_at = now + timedelta(seconds=max(self.lease_seconds, int(job.max_runtime_seconds) + 60))
        row = self.store.lease_job(job.job_id, worker_id, lease_token, lease_expires_at)
        leased_job = self._job_from_row(row)
        return DiagnosticLease(
            job=leased_job,
            worker_id=worker_id,
            lease_token=lease_token,
            lease_expires_at=lease_expires_at,
            attempt_number=int(row["attempt_count"]),
        )

    def complete(self, lease: DiagnosticLease, result: DiagnosticResult) -> DiagnosticResult:
        if lease.job.job_id != result.job_id:
            raise DiagnosticLeaseError("result does not match the leased job")
        self.store.complete_job(lease.job.job_id, lease.lease_token, result)
        return result

    def heartbeat(self, worker_id: str, status: str, metadata: Mapping[str, Any] | None = None) -> None:
        self.store.record_heartbeat(worker_id, status, metadata)


class ExecutionRunner(Protocol):
    def execute(self, lease: DiagnosticLease) -> DiagnosticResult:
        raise NotImplementedError


class MockSqlclFixtureRunner:
    """Deterministic SQLcl fixture runner for acceptance tests."""

    def __init__(self, fixtures: Mapping[str, SqlclFixture]):
        self.fixtures = dict(fixtures)
        self.execution_count: dict[str, int] = {}

    def _fixture_key(self, lease: DiagnosticLease) -> str:
        return lease.job.template_id or lease.job.job_id

    def execute(self, lease: DiagnosticLease) -> DiagnosticResult:
        job = lease.job
        fixture = self.fixtures.get(self._fixture_key(lease))
        if fixture is None:
            raise KeyError(f"missing SQLcl fixture for {self._fixture_key(lease)}")
        if fixture.exit_code != 0:
            raise RuntimeError(f"fixture reports non-zero exit code: {fixture.exit_code}")
        if fixture.runtime_seconds > job.max_runtime_seconds:
            raise TimeoutError("diagnostic run exceeded max_runtime_seconds")

        rendered_stdout = fixture.stdout
        row_count = 0
        truncated = False
        if fixture.output_format == "csv":
            lines = [line for line in rendered_stdout.splitlines() if line.strip()]
            if lines:
                header, *rows = lines
                row_count = len(rows)
                if row_count > job.max_rows:
                    rows = rows[: job.max_rows]
                    row_count = len(rows)
                    truncated = True
                rendered_stdout = "\n".join([header, *rows])
        else:
            rows = [line for line in rendered_stdout.splitlines() if line.strip()]
            row_count = len(rows)
            if row_count > job.max_rows:
                rows = rows[: job.max_rows]
                row_count = len(rows)
                rendered_stdout = "\n".join(rows)
                truncated = True

        stdout_bytes = rendered_stdout.encode("utf-8")
        if len(stdout_bytes) > job.max_output_bytes:
            stdout_bytes = stdout_bytes[: job.max_output_bytes]
            rendered_stdout = stdout_bytes.decode("utf-8", errors="ignore")
            truncated = True

        stderr = fixture.stderr
        stdout_hash = _sha256_text(rendered_stdout)
        stderr_hash = _sha256_text(stderr)
        output_hash = _sha256_text(
            canonical_json(
                {
                    "stdout_hash": stdout_hash,
                    "stderr_hash": stderr_hash,
                    "exit_code": fixture.exit_code,
                    "row_count": row_count,
                    "truncated": truncated,
                    "sql_text": job.sql_text,
                }
            )
        )
        result_hash = _sha256_text(
            canonical_json(
                {
                    "job_id": job.job_id,
                    "lease_token": lease.lease_token,
                    "worker_id": lease.worker_id,
                    "output_hash": output_hash,
                    "sql_text_hash": job.payload_hash,
                }
            )
        )
        self.execution_count[job.job_id] = self.execution_count.get(job.job_id, 0) + 1
        return DiagnosticResult(
            job_id=job.job_id,
            lease_token=lease.lease_token,
            worker_id=lease.worker_id,
            stdout=rendered_stdout,
            stderr=stderr,
            exit_code=fixture.exit_code,
            row_count=row_count,
            runtime_seconds=fixture.runtime_seconds,
            truncated=truncated,
            stdout_hash=stdout_hash,
            stderr_hash=stderr_hash,
            output_hash=output_hash,
            result_hash=result_hash,
            metadata={"fixture_key": self._fixture_key(lease), "sql_text": job.sql_text},
        )


@dataclass
class QueryOutputProvenanceHook:
    platform: KnowledgePlatform

    def ingest(self, lease: DiagnosticLease, result: DiagnosticResult):
        job = lease.job
        metadata = {
            "job_id": job.job_id,
            "lease_token": lease.lease_token,
            "result_hash": result.result_hash,
            "stdout_hash": result.stdout_hash,
            "stderr_hash": result.stderr_hash,
            "output_hash": result.output_hash,
            "audit_id": job.audit_id,
            "finding_id": job.finding_id,
            "template_id": job.template_id,
            "parameters": job.parameters,
            "operation_class": job.operation_class,
            "database_profile_alias": job.database_profile_alias,
            "wrapper_name": job.wrapper_name,
            "row_count": result.row_count,
            "truncated": result.truncated,
            "exit_code": result.exit_code,
        }
        return self.platform.ingest(
            source_id=f"diagnostic-{job.job_id}",
            source_uri=f"broker://{job.database_profile_alias}/{job.job_id}",
            content=result.stdout or result.stderr or "",
            mime_type="text/csv" if job.expected_output_format == "csv" else "text/plain",
            source_kind="diagnostic-output",
            channel="onprem-worker",
            domain_id=job.audit_id or "diagnostics",
            system_id=job.database_profile_alias,
            component_id="sqlcl-wrapper",
            evidence_type="query-output",
            acl_scope="internal",
            domain=job.audit_id or "diagnostics",
            system=job.database_profile_alias,
            source_type="query-output",
            parser_name="sqlcl",
            parser_version="1",
            namespace_label=f"diagnostic::{job.database_profile_alias}",
            metadata=metadata,
        )


class DiagnosticWorker:
    """Poll-only worker with offline spool and restartability."""

    def __init__(
        self,
        broker: SecureDiagnosticBroker,
        runner: ExecutionRunner,
        spool_dir: str | Path,
        *,
        worker_id: str | None = None,
        provenance_hook: QueryOutputProvenanceHook | None = None,
    ):
        self.broker = broker
        self.runner = runner
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self.spool_dir = Path(spool_dir)
        self.spool_dir.mkdir(parents=True, exist_ok=True)
        self.provenance_hook = provenance_hook

    def _spool_path(self, job_id: str) -> Path:
        return self.spool_dir / f"{job_id}.json"

    def _write_spool(self, lease: DiagnosticLease, *, result: DiagnosticResult | None = None, submitted: bool = False) -> None:
        payload = {
            "lease": {
                "worker_id": lease.worker_id,
                "lease_token": lease.lease_token,
                "lease_expires_at": lease.lease_expires_at.isoformat(),
                "attempt_number": lease.attempt_number,
            },
            "job": lease.job.to_dict(),
            "result": result.to_dict() if result else None,
            "submitted": submitted,
            "state": "executed" if result else "claimed",
        }
        self._spool_path(lease.job.job_id).write_text(canonical_json(payload), encoding="utf-8")

    def _load_spool(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _lease_from_spool(self, data: Mapping[str, Any]) -> DiagnosticLease:
        job_data = dict(data["job"])
        lease_data = dict(data["lease"])
        job_data["created_at"] = datetime.fromisoformat(job_data["created_at"])
        job_data["expires_at"] = datetime.fromisoformat(job_data["expires_at"])
        job = DiagnosticJob(**job_data)
        return DiagnosticLease(
            job=job,
            worker_id=lease_data["worker_id"],
            lease_token=lease_data["lease_token"],
            lease_expires_at=datetime.fromisoformat(lease_data["lease_expires_at"]),
            attempt_number=int(lease_data["attempt_number"]),
        )

    def _result_from_spool(self, data: Mapping[str, Any]) -> DiagnosticResult | None:
        if not data.get("result"):
            return None
        return DiagnosticResult(**data["result"])

    def _recover_spool(self) -> DiagnosticResult | None:
        recovered: DiagnosticResult | None = None
        for path in sorted(self.spool_dir.glob("*.json")):
            data = self._load_spool(path)
            lease = self._lease_from_spool(data)
            result = self._result_from_spool(data)
            submitted = bool(data.get("submitted"))
            if result is None:
                result = self.runner.execute(lease)
                self._write_spool(lease, result=result, submitted=False)
            if not submitted:
                try:
                    self.broker.complete(lease, result)
                except DiagnosticReplayError:
                    pass
                if self.provenance_hook is not None:
                    self.provenance_hook.ingest(lease, result)
                self._write_spool(lease, result=result, submitted=True)
            path.unlink(missing_ok=True)
            recovered = result
        return recovered

    def run_once(self) -> DiagnosticResult | None:
        recovered = self._recover_spool()
        if recovered is not None:
            self.broker.heartbeat(self.worker_id, "healthy", {"recovered": True, "job_id": recovered.job_id})
            return recovered

        lease = self.broker.poll(self.worker_id)
        if lease is None:
            self.broker.heartbeat(self.worker_id, "idle", {"spool_size": len(list(self.spool_dir.glob("*.json")))})
            return None
        self._write_spool(lease)
        result = self.runner.execute(lease)
        self._write_spool(lease, result=result, submitted=False)
        try:
            self.broker.complete(lease, result)
        except DiagnosticReplayError:
            pass
        if self.provenance_hook is not None:
            self.provenance_hook.ingest(lease, result)
        self._write_spool(lease, result=result, submitted=True)
        self._spool_path(lease.job.job_id).unlink(missing_ok=True)
        self.broker.heartbeat(self.worker_id, "healthy", {"job_id": lease.job.job_id, "result_hash": result.result_hash})
        return result


__all__ = [
    "APPROVED_WRAPPER",
    "ALLOWED_OPERATION_CLASSES",
    "DiagnosticBrokerStore",
    "DiagnosticJob",
    "DiagnosticLease",
    "DiagnosticLeaseError",
    "DiagnosticPolicyError",
    "DiagnosticReplayError",
    "DiagnosticResult",
    "DiagnosticTemplate",
    "DiagnosticTemplateRegistry",
    "DiagnosticWorker",
    "MockSqlclFixtureRunner",
    "QueryOutputProvenanceHook",
    "SecureDiagnosticBroker",
    "SqlclFixture",
    "canonical_json",
    "validate_read_only_sql",
]
