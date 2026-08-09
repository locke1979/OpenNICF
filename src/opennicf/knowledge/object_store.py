"""Object storage abstractions for immutable evidence artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol
import json
import os
import tempfile


@dataclass(frozen=True)
class ObjectReference:
    backend: str
    content_hash: str
    object_key: str
    size_bytes: int
    mime_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


class ObjectStore(Protocol):
    backend_name: str

    def put_bytes(self, content: bytes, *, mime_type: str, metadata: dict[str, Any] | None = None) -> ObjectReference:
        raise NotImplementedError

    def get_bytes(self, object_key: str) -> bytes:
        raise NotImplementedError

    def exists(self, object_key: str) -> bool:
        raise NotImplementedError


class FilesystemObjectStore:
    backend_name = "filesystem"

    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, content_hash: str) -> Path:
        return self.root / content_hash[:2] / content_hash

    def put_bytes(self, content: bytes, *, mime_type: str, metadata: dict[str, Any] | None = None) -> ObjectReference:
        content_hash = sha256(content).hexdigest()
        target = self._path_for(content_hash)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
                handle.write(content)
                temp_name = Path(handle.name)
            temp_name.replace(target)
        return ObjectReference(
            backend=self.backend_name,
            content_hash=content_hash,
            object_key=str(target.relative_to(self.root)),
            size_bytes=len(content),
            mime_type=mime_type,
            metadata=dict(metadata or {}),
        )

    def get_bytes(self, object_key: str) -> bytes:
        return (self.root / object_key).read_bytes()

    def exists(self, object_key: str) -> bool:
        return (self.root / object_key).exists()


class MemoryObjectStore:
    backend_name = "memory"

    def __init__(self):
        self._objects: dict[str, bytes] = {}
        self._mime_types: dict[str, str] = {}

    def put_bytes(self, content: bytes, *, mime_type: str, metadata: dict[str, Any] | None = None) -> ObjectReference:
        content_hash = sha256(content).hexdigest()
        self._objects.setdefault(content_hash, content)
        self._mime_types[content_hash] = mime_type
        return ObjectReference(
            backend=self.backend_name,
            content_hash=content_hash,
            object_key=content_hash,
            size_bytes=len(content),
            mime_type=mime_type,
            metadata=dict(metadata or {}),
        )

    def get_bytes(self, object_key: str) -> bytes:
        return self._objects[object_key]

    def exists(self, object_key: str) -> bool:
        return object_key in self._objects

    def snapshot(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "objects": {
                key: {
                    "content": value.decode("utf-8", errors="surrogateescape"),
                    "mime_type": self._mime_types.get(key, "application/octet-stream"),
                }
                for key, value in self._objects.items()
            },
        }

    @classmethod
    def restore(cls, snapshot: dict[str, Any]) -> "MemoryObjectStore":
        store = cls()
        for key, data in snapshot.get("objects", {}).items():
            content = data.get("content", "").encode("utf-8", errors="surrogateescape")
            store._objects[key] = content
            store._mime_types[key] = data.get("mime_type", "application/octet-stream")
        return store

    def to_json(self) -> str:
        return json.dumps(self.snapshot(), sort_keys=True)

