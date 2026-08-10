"""Restart-safe code symbol extraction with an optional Tree-sitter front end."""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable
from hashlib import sha256
from typing import Any

from .models import ChunkRecord, CodeRelationshipRecord, CodeSymbolRecord

_DECLARATION_PATTERNS = {
    "python": (("class", re.compile(r"^\s*class\s+([A-Za-z_]\w*)")), ("function", re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)"))),
    "java": (("class", re.compile(r"^\s*(?:public\s+|private\s+|protected\s+|final\s+|abstract\s+)*class\s+([A-Za-z_]\w*)")), ("method", re.compile(r"^\s*(?:public|private|protected|static|final|native|synchronized|abstract|\s)+[\w<>\[\], ?]+\s+([A-Za-z_]\w*)\s*\("))),
    "sql": (("table", re.compile(r"\bCREATE\s+(?:TEMP(?:ORARY)?\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([\w.]+)", re.IGNORECASE)), ("routine", re.compile(r"\bCREATE\s+(?:OR\s+REPLACE\s+)?(?:FUNCTION|PROCEDURE)\s+([\w.]+)", re.IGNORECASE))),
    "powershell": (("function", re.compile(r"^\s*function\s+([\w.-]+)", re.IGNORECASE)),),
}
_EXTENSIONS = {".py": "python", ".java": "java", ".sql": "sql", ".ps1": "powershell", ".psm1": "powershell"}
_IMPORT_RE = re.compile(r"^\s*(?:import|from)\s+([\w.:-]+)", re.IGNORECASE | re.MULTILINE)
_CALL_RE = re.compile(r"\b([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?)\s*\(")
_ENDPOINT_RE = re.compile(r"(?:@(?:app|router)\.(?:get|post|put|delete)|(?:GET|POST|PUT|DELETE)\s+)(?:\(|\s+)?[\"']([^\"']+)", re.IGNORECASE)
_CONFIG_RE = re.compile(r"(?:os\.environ(?:\.get)?|ENV|Get-Item\s+env:|Configuration(?:Manager)?[.]AppSettings)\s*[(\[]?\s*[\"']([A-Z][A-Z0-9_.-]+)", re.IGNORECASE)


def _language(source_uri: str) -> str | None:
    return _EXTENSIONS.get("." + source_uri.rsplit(".", 1)[-1].lower()) if "." in source_uri else None


def _tree_sitter_language(language: str) -> Any | None:
    """Return a parser when deployment extras are installed; never fail indexing."""
    try:
        from tree_sitter_language_pack import get_parser
        return get_parser(language)
    except Exception:  # noqa: BLE001 - optional parser boundary
        return None


def _stable_id(prefix: str, *parts: str) -> str:
    return f"{prefix}_{sha256(chr(0).join(parts).encode()).hexdigest()[:32]}"


def build_code_index(content: str, *, source: Any, version: Any, artifact: Any, chunks: Iterable[ChunkRecord]) -> tuple[tuple[CodeSymbolRecord, ...], tuple[CodeRelationshipRecord, ...]]:
    """Extract bounded symbols and relationships for one immutable source version."""
    language = _language(source.source_uri)
    if not language:
        return (), ()
    parser = _tree_sitter_language(language)
    parser_name = "tree-sitter" if parser is not None else "fallback"
    if parser is not None:
        try:
            parser.parse(content.encode("utf-8"))
        except Exception:  # noqa: BLE001 - malformed/unsupported syntax is safe
            parser_name = "fallback"
    elif language == "python":
        try:
            ast.parse(content)
        except (SyntaxError, ValueError):
            pass
    lines = content.splitlines()
    symbols: list[CodeSymbolRecord] = []
    declarations = _DECLARATION_PATTERNS[language]
    for line_number, line in enumerate(lines, 1):
        for kind, pattern in declarations:
            match = pattern.search(line)
            if not match:
                continue
            name = match.group(1)
            qualified = name
            if kind in {"function", "method"}:
                parents = [item.name for item in symbols if item.kind == "class" and item.line_start < line_number]
                qualified = f"{parents[-1]}.{name}" if parents else name
            chunk = next((item for item in chunks if (item.line_start or 1) <= line_number <= (item.line_end or line_number)), None)
            if chunk is None:
                continue
            symbol_id = _stable_id("sym", version.source_version_id, qualified, str(line_number))
            symbols.append(CodeSymbolRecord(symbol_id, source.source_id, version.source_version_id, artifact.artifact_hash, chunk.chunk_id, name, qualified, kind, line.strip(), f"{chunk.locator}:L{line_number}", line_number, line_number, parser_name, version.parser_version, version.content_hash, source.namespace_id, source.domain_id, source.system_id, source.component_id, source.environment, source.evidence_type, source.acl_scope, {"language": language}))
    known = {item.name for item in symbols} | {item.qualified_name for item in symbols}
    relationships: list[CodeRelationshipRecord] = []
    for symbol in symbols:
        body = "\n".join(lines[symbol.line_start - 1 : min(len(lines), symbol.line_end + 80)])
        refs: list[tuple[str, str]] = [(name, "imports") for name in _IMPORT_RE.findall(body)]
        refs.extend((name, "calls") for name in _CALL_RE.findall(body) if name.split(".")[-1] in known)
        refs.extend((name, "endpoint") for name in _ENDPOINT_RE.findall(body))
        refs.extend((name, "configuration") for name in _CONFIG_RE.findall(body))
        if re.search(r"\b(?:logger|logging)\.(?:debug|info|warning|error|exception|critical)\b|Write-(?:Verbose|Warning|Error|Information)", body, re.IGNORECASE):
            refs.append(("logging", "logging"))
        if re.search(r"\b(?:try|catch|except|finally)\b", body, re.IGNORECASE):
            refs.append(("exception-handling", "exception_handling"))
        for target, relation in dict.fromkeys(refs):
            relationships.append(CodeRelationshipRecord(_stable_id("rel", symbol.symbol_id, relation, target), symbol.symbol_id, target, relation, symbol.source_id, symbol.source_version_id, symbol.artifact_hash, symbol.chunk_id, symbol.locator, symbol.parser_name, symbol.parser_version, symbol.source_hash, symbol.namespace_id, symbol.domain_id, symbol.system_id, symbol.component_id, symbol.environment, symbol.evidence_type, symbol.acl_scope, {"language": language, "resolved": target in known}))
    return tuple(symbols), tuple(relationships)
