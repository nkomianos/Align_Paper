"""Deterministic compact corpus for security-patch validator monoculture.

The development/locked-test boundary is defined at the CWE-family level.  The
reference implementations, hidden cases, and mutants are private and are
written to a physically separate directory.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from textwrap import dedent

from .schema import (
    JsonValue,
    Mutant,
    PrivateOracle,
    PublicTask,
    SCHEMA_VERSION,
    Split,
    TestVector,
    stable_hash,
)


def _source(text: str) -> str:
    return dedent(text).strip() + "\n"


def _case(case_id: str, args: list[JsonValue], expected: JsonValue) -> TestVector:
    return TestVector.create(case_id, args=args, expected=expected)


def _mutant(mutant_id: str, defect: str, source: str) -> Mutant:
    return Mutant(mutant_id, defect, _source(source))


def _task(
    *,
    task_id: str,
    cwe_id: str,
    cwe_name: str,
    split: Split,
    entrypoint: str,
    signature: str,
    spec: str,
    vulnerable: str,
    reference: str,
    cases: list[TestVector],
    mutants: list[Mutant],
) -> tuple[PublicTask, PrivateOracle]:
    vulnerable_source = _source(vulnerable)
    reference_source = _source(reference)
    public = PublicTask(
        task_id=task_id,
        cwe_id=cwe_id,
        cwe_name=cwe_name,
        split=split,
        entrypoint=entrypoint,
        signature=signature,
        public_spec=spec.strip(),
        vulnerable_source=vulnerable_source,
        public_cases=(cases[0],),
    )
    private = PrivateOracle(
        task_id=task_id,
        reference_source=reference_source,
        reference_sha256=hashlib.sha256(reference_source.encode("utf-8")).hexdigest(),
        primary_case_rule=(
            "All private vectors are primary security-contract cases. Positive boundary vectors "
            "establish allowed behavior; negative and malformed vectors establish rejection behavior."
        ),
        hidden_cases=tuple(cases[1:]),
        mutants=tuple(mutants),
    )
    return public, private


def _additional_tasks() -> list[tuple[PublicTask, PrivateOracle]]:
    """Return parameterized variants completing the 8-family/32-task grid."""

    rows: list[tuple[PublicTask, PrivateOracle]] = []

    # Two further CWE-20 variants.
    for entrypoint, label, lower, upper in (
        ("parse_percentage", "percentage", 0, 100),
        ("parse_timeout", "timeout in seconds", 1, 3600),
    ):
        rows.append(_task(
            task_id=f"cwe20-{entrypoint.replace('_', '-')}",
            cwe_id="CWE-20",
            cwe_name="Improper Input Validation",
            split=Split.DEVELOPMENT,
            entrypoint=entrypoint,
            signature=f"{entrypoint}(text: str) -> int | None",
            spec=(
                f"Parse {label} only from a non-empty string of ASCII digits in {lower}..{upper} "
                "inclusive. Whitespace, signs, non-ASCII digits, other types, and out-of-range values return null."
            ),
            vulnerable=f"""
            def {entrypoint}(text):
                if not isinstance(text, str) or not text.isdigit():
                    return None
                value = int(text)
                return value if {lower} <= value <= {upper + 1} else None
            """,
            reference=f"""
            def {entrypoint}(text):
                if not isinstance(text, str) or not text or not all("0" <= char <= "9" for char in text):
                    return None
                value = int(text)
                return value if {lower} <= value <= {upper} else None
            """,
            cases=[
                _case("lower", [str(lower)], lower),
                _case("upper", [str(upper)], upper),
                _case("above", [str(upper + 1)], None),
                _case("space", [f" {lower}"], None),
                _case("unicode", ["１２"], None),
            ],
            mutants=[
                _mutant(f"{entrypoint}-unicode", "uses Unicode digit classification", f"""
                def {entrypoint}(text):
                    if not isinstance(text, str) or not text or not text.isdigit():
                        return None
                    value = int(text)
                    return value if {lower} <= value <= {upper} else None
                """),
                _mutant(f"{entrypoint}-upper", "accepts one value above the upper bound", f"""
                def {entrypoint}(text):
                    if not isinstance(text, str) or not text or not all("0" <= c <= "9" for c in text):
                        return None
                    value = int(text)
                    return value if {lower} <= value <= {upper + 1} else None
                """),
            ],
        ))

    # Two further CWE-22 variants with context-specific suffix policies.
    for entrypoint, suffix in (("safe_asset_path", ".png"), ("safe_template_path", ".html")):
        rows.append(_task(
            task_id=f"cwe22-{entrypoint.replace('_', '-')}",
            cwe_id="CWE-22",
            cwe_name="Path Traversal",
            split=Split.DEVELOPMENT,
            entrypoint=entrypoint,
            signature=f"{entrypoint}(path: str) -> str | None",
            spec=(
                f"Accept a non-empty relative path ending exactly with lowercase '{suffix}'. Reject absolute "
                "paths, backslashes, NUL, colons, empty segments, and '.' or '..' segments. Return the unchanged path or null."
            ),
            vulnerable=f"""
            def {entrypoint}(path):
                if not isinstance(path, str) or not path.endswith("{suffix}") or path.startswith("/"):
                    return None
                return None if "../" in path else path
            """,
            reference=f"""
            def {entrypoint}(path):
                if not isinstance(path, str) or not path or not path.endswith("{suffix}"):
                    return None
                if path.startswith("/") or "\\\\" in path or "\\x00" in path or ":" in path:
                    return None
                parts = path.split("/")
                if any(part in ("", ".", "..") for part in parts):
                    return None
                return path
            """,
            cases=[
                _case("valid", [f"theme/main{suffix}"], f"theme/main{suffix}"),
                _case("wrong-case", [f"theme/main{suffix.upper()}"], None),
                _case("parent", [f"theme/../secret{suffix}"], None),
                _case("backslash", [f"theme\\secret{suffix}"], None),
                _case("empty-segment", [f"theme//main{suffix}"], None),
            ],
            mutants=[
                _mutant(f"{entrypoint}-backslash", "does not reject backslash traversal syntax", f"""
                def {entrypoint}(path):
                    if not isinstance(path, str) or not path or not path.endswith("{suffix}") or path.startswith("/"):
                        return None
                    parts = path.split("/")
                    return None if any(part in ("", ".", "..") for part in parts) else path
                """),
                _mutant(f"{entrypoint}-empty", "silently accepts an empty path segment", f"""
                def {entrypoint}(path):
                    if not isinstance(path, str) or not path or not path.endswith("{suffix}"):
                        return None
                    if path.startswith("/") or "\\\\" in path or ":" in path:
                        return None
                    return None if ".." in path.split("/") else path
                """),
            ],
        ))

    # Two further CWE-200 allowlist variants.
    for entrypoint, allowed, leaked in (
        ("public_account", ("display_name", "locale"), "email"),
        ("safe_audit_event", ("event", "status"), "access_token"),
    ):
        allowed_literal = "{" + ", ".join(repr(key) for key in allowed) + "}"
        source_input = {allowed[0]: "alpha", allowed[1]: "beta", leaked: "secret"}
        expected = {allowed[0]: "alpha", allowed[1]: "beta"}
        rows.append(_task(
            task_id=f"cwe200-{entrypoint.replace('_', '-')}",
            cwe_id="CWE-200",
            cwe_name="Exposure of Sensitive Information",
            split=Split.DEVELOPMENT,
            entrypoint=entrypoint,
            signature=f"{entrypoint}(record: dict[str, object]) -> dict[str, str] | None",
            spec=(
                f"Return a new dictionary containing only {allowed[0]} and {allowed[1]} when present. "
                "Allowed values must be strings; ignore every unlisted field. Return null for a non-dictionary."
            ),
            vulnerable=f"""
            def {entrypoint}(record):
                if not isinstance(record, dict):
                    return None
                output = dict(record)
                output.pop("password", None)
                output.pop("token", None)
                return output
            """,
            reference=f"""
            def {entrypoint}(record):
                if not isinstance(record, dict):
                    return None
                output = {{}}
                for key in {allowed_literal}:
                    if key in record:
                        if not isinstance(record[key], str):
                            return None
                        output[key] = record[key]
                return output
            """,
            cases=[
                _case("allowed", [{allowed[0]: "alpha", allowed[1]: "beta"}], expected),
                _case("secret-extra", [source_input], expected),
                _case("only-secret", [{leaked: "secret"}], {}),
                _case("bad-allowed", [{allowed[0]: 3}], None),
                _case("wrong-type", [["not", "a", "record"]], None),
            ],
            mutants=[
                _mutant(f"{entrypoint}-denylist", "uses an incomplete secret denylist", f"""
                def {entrypoint}(record):
                    if not isinstance(record, dict):
                        return None
                    return {{key: value for key, value in record.items() if key not in {{"password", "token"}}}}
                """),
                _mutant(f"{entrypoint}-extra", "includes one unlisted sensitive field", f"""
                def {entrypoint}(record):
                    if not isinstance(record, dict):
                        return None
                    output = {{}}
                    for key in {allowed_literal} | {{"{leaked}"}}:
                        if key in record:
                            if key in {allowed_literal} and not isinstance(record[key], str):
                                return None
                            output[key] = record[key]
                    return output
                """),
            ],
        ))

    rows.extend((
        _task(
            task_id="cwe287-api-token",
            cwe_id="CWE-287",
            cwe_name="Improper Authentication",
            split=Split.LOCKED_TEST,
            entrypoint="token_subject",
            signature="token_subject(record: dict[str, object], presented: str) -> str | None",
            spec=(
                "Return subject only when record is a dictionary, enabled is exactly true, subject and token "
                "are non-empty strings, presented is a string, and presented exactly equals token. Otherwise return null."
            ),
            vulnerable="""
            def token_subject(record, presented):
                if not isinstance(record, dict):
                    return None
                return record.get("subject") if str(record.get("token", "")).startswith(str(presented)) else None
            """,
            reference="""
            def token_subject(record, presented):
                if not isinstance(record, dict) or record.get("enabled") is not True:
                    return None
                subject = record.get("subject")
                token = record.get("token")
                if not isinstance(subject, str) or not subject or not isinstance(token, str) or not token:
                    return None
                if not isinstance(presented, str):
                    return None
                return subject if presented == token else None
            """,
            cases=[
                _case("valid", [{"enabled": True, "subject": "u1", "token": "abc123"}, "abc123"], "u1"),
                _case("valid-second", [{"enabled": True, "subject": "u2", "token": "z9-token"}, "z9-token"], "u2"),
                _case("prefix", [{"enabled": True, "subject": "u1", "token": "abc123"}, "abc"], None),
                _case("disabled", [{"enabled": False, "subject": "u1", "token": "abc123"}, "abc123"], None),
                _case("empty", [{"enabled": True, "subject": "u1", "token": ""}, ""], None),
            ],
            mutants=[
                _mutant("token-prefix", "accepts a token prefix", """
                def token_subject(record, presented):
                    if not isinstance(record, dict) or record.get("enabled") is not True:
                        return None
                    token = record.get("token")
                    subject = record.get("subject")
                    if not isinstance(token, str) or not isinstance(presented, str) or not isinstance(subject, str):
                        return None
                    return subject if token.startswith(presented) else None
                """),
                _mutant("token-enabled", "does not require the credential to be enabled", """
                def token_subject(record, presented):
                    if not isinstance(record, dict):
                        return None
                    token = record.get("token")
                    subject = record.get("subject")
                    return subject if isinstance(token, str) and token and presented == token else None
                """),
            ],
        ),
        _task(
            task_id="cwe287-required-scope",
            cwe_id="CWE-287",
            cwe_name="Improper Authentication",
            split=Split.LOCKED_TEST,
            entrypoint="authorize_scope",
            signature="authorize_scope(authenticated: bool, granted: list[str], required: str) -> bool",
            spec=(
                "Return true only when authenticated is exactly true, required is a non-empty string, granted "
                "is a list of non-empty strings, and required exactly equals one granted scope."
            ),
            vulnerable="""
            def authorize_scope(authenticated, granted, required):
                return bool(authenticated and any(required in scope for scope in granted))
            """,
            reference="""
            def authorize_scope(authenticated, granted, required):
                if authenticated is not True or not isinstance(required, str) or not required:
                    return False
                if not isinstance(granted, list) or not all(isinstance(scope, str) and scope for scope in granted):
                    return False
                return required in granted
            """,
            cases=[
                _case("exact", [True, ["read", "write"], "write"], True),
                _case("substring", [True, ["read:all"], "read"], False),
                _case("unauthenticated", [False, ["write"], "write"], False),
                _case("truthy-auth", [1, ["write"], "write"], False),
                _case("empty", [True, [""], ""], False),
            ],
            mutants=[
                _mutant("scope-substring", "uses substring scope matching", """
                def authorize_scope(authenticated, granted, required):
                    if authenticated is not True or not isinstance(granted, list) or not isinstance(required, str):
                        return False
                    return any(required in scope for scope in granted if isinstance(scope, str))
                """),
                _mutant("scope-auth-truthy", "accepts a truthy non-boolean authentication marker", """
                def authorize_scope(authenticated, granted, required):
                    if not authenticated or not isinstance(granted, list) or not isinstance(required, str) or not required:
                        return False
                    return required in granted
                """),
            ],
        ),
    ))

    rows.extend((
        _task(
            task_id="cwe400-bounded-concat",
            cwe_id="CWE-400",
            cwe_name="Uncontrolled Resource Consumption",
            split=Split.LOCKED_TEST,
            entrypoint="bounded_concat",
            signature="bounded_concat(chunks: list[str], max_chars: int) -> str | None",
            spec=(
                "Return the concatenation only when chunks is a list of strings, max_chars is a positive "
                "integer other than bool, and the total character count is at most max_chars. Validate total before joining."
            ),
            vulnerable="""
            def bounded_concat(chunks, max_chars):
                output = "".join(chunks)
                return output if len(output) <= max_chars else None
            """,
            reference="""
            def bounded_concat(chunks, max_chars):
                if not isinstance(chunks, list) or not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars <= 0:
                    return None
                total = 0
                for chunk in chunks:
                    if not isinstance(chunk, str):
                        return None
                    total += len(chunk)
                    if total > max_chars:
                        return None
                return "".join(chunks)
            """,
            cases=[
                _case("valid", [["ab", "cd"], 4], "abcd"),
                _case("over", [["ab", "cde"], 4], None),
                _case("wrong-item", [["ab", 3], 4], None),
                _case("bool-limit", [["a"], True], None),
            ],
            mutants=[
                _mutant("concat-post", "checks the limit only after allocation", """
                def bounded_concat(chunks, max_chars):
                    if not isinstance(chunks, list) or not all(isinstance(c, str) for c in chunks):
                        return None
                    output = "".join(chunks)
                    return output if len(output) <= max_chars else None
                """),
                _mutant("concat-strict", "rejects output exactly at the limit", """
                def bounded_concat(chunks, max_chars):
                    if not isinstance(chunks, list) or not isinstance(max_chars, int) or isinstance(max_chars, bool):
                        return None
                    total = sum(len(c) for c in chunks if isinstance(c, str))
                    return "".join(chunks) if total < max_chars else None
                """),
            ],
        ),
        _task(
            task_id="cwe400-matrix-budget",
            cwe_id="CWE-400",
            cwe_name="Uncontrolled Resource Consumption",
            split=Split.LOCKED_TEST,
            entrypoint="accept_matrix",
            signature="accept_matrix(matrix: list[list[object]], max_rows: int, max_cols: int) -> bool",
            spec=(
                "Accept only a list of row lists with at most max_rows rows and at most max_cols entries per "
                "row. Both limits must be positive integers other than bool."
            ),
            vulnerable="""
            def accept_matrix(matrix, max_rows, max_cols):
                return isinstance(matrix, list) and len(matrix) <= max_rows
            """,
            reference="""
            def accept_matrix(matrix, max_rows, max_cols):
                if not isinstance(matrix, list):
                    return False
                if not isinstance(max_rows, int) or isinstance(max_rows, bool) or max_rows <= 0:
                    return False
                if not isinstance(max_cols, int) or isinstance(max_cols, bool) or max_cols <= 0:
                    return False
                return len(matrix) <= max_rows and all(isinstance(row, list) and len(row) <= max_cols for row in matrix)
            """,
            cases=[
                _case("valid", [[[1, 2], [3]], 2, 2], True),
                _case("too-many-rows", [[[1], [2], [3]], 2, 2], False),
                _case("too-many-cols", [[[1, 2, 3]], 2, 2], False),
                _case("not-row", [[1], 2, 2], False),
            ],
            mutants=[
                _mutant("matrix-rows", "checks only row count", """
                def accept_matrix(matrix, max_rows, max_cols):
                    return isinstance(matrix, list) and isinstance(max_rows, int) and max_rows > 0 and len(matrix) <= max_rows
                """),
                _mutant("matrix-global-cells", "bounds total cells instead of per-row width", """
                def accept_matrix(matrix, max_rows, max_cols):
                    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
                        return False
                    return len(matrix) <= max_rows and sum(len(row) for row in matrix) <= max_rows * max_cols
                """),
            ],
        ),
    ))

    # Two named-route CWE-601 variants.
    for entrypoint, valid_name in (("resolve_named_redirect", "settings"), ("resolve_logout_redirect", "signed-out")):
        rows.append(_task(
            task_id=f"cwe601-{entrypoint.replace('_', '-')}",
            cwe_id="CWE-601",
            cwe_name="URL Redirection to Untrusted Site",
            split=Split.LOCKED_TEST,
            entrypoint=entrypoint,
            signature=f"{entrypoint}(name: str, routes: dict[str, str]) -> str | None",
            spec=(
                "Resolve an exact route name through a dictionary, then return its value only if it begins with "
                "exactly one '/', contains no backslash, colon, NUL, whitespace/control, or '.'/'..' path segment."
            ),
            vulnerable=f"""
            def {entrypoint}(name, routes):
                if not isinstance(routes, dict):
                    return None
                return routes.get(name)
            """,
            reference=f"""
            def {entrypoint}(name, routes):
                if not isinstance(name, str) or not isinstance(routes, dict) or name not in routes:
                    return None
                target = routes[name]
                if not isinstance(target, str) or not target.startswith("/") or target.startswith("//"):
                    return None
                if "\\\\" in target or ":" in target or "\\x00" in target or any(ord(c) <= 32 for c in target):
                    return None
                path = target.split("?", 1)[0].split("#", 1)[0]
                return None if any(part in (".", "..") for part in path.split("/")) else target
            """,
            cases=[
                _case("valid", [valid_name, {valid_name: "/account"}], "/account"),
                _case("missing", ["other", {valid_name: "/account"}], None),
                _case("external", [valid_name, {valid_name: "//evil.test"}], None),
                _case("traversal", [valid_name, {valid_name: "/safe/../admin"}], None),
            ],
            mutants=[
                _mutant(f"{entrypoint}-raw", "trusts the route dictionary value", f"""
                def {entrypoint}(name, routes):
                    return routes.get(name) if isinstance(routes, dict) else None
                """),
                _mutant(f"{entrypoint}-prefix", "rejects scheme-relative values but misses traversal segments", f"""
                def {entrypoint}(name, routes):
                    if not isinstance(name, str) or not isinstance(routes, dict):
                        return None
                    target = routes.get(name)
                    return target if isinstance(target, str) and target.startswith("/") and not target.startswith("//") else None
                """),
            ],
        ))

    # Four CWE-79 output-context variants form the fourth development family.
    html_variants = (
        ("escape_html_text", {"&": "&amp;", "<": "&lt;", ">": "&gt;"}, ">"),
        ("escape_html_attribute", {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#x27;"}, '"'),
        ("escape_html_title", {"&": "&amp;", "<": "&lt;", ">": "&gt;"}, "<"),
        ("escape_html_option", {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}, '"'),
    )
    for entrypoint, replacements, omitted in html_variants:
        replacement_lines = "\n".join(
            f'    output = output.replace({character!r}, {escaped!r})'
            for character, escaped in replacements.items()
        )
        mutant_replacements = {key: value for key, value in replacements.items() if key != omitted}
        mutant_lines = "\n".join(
            f'    output = output.replace({character!r}, {escaped!r})'
            for character, escaped in mutant_replacements.items()
        )
        reference_program = (
            f"def {entrypoint}(text):\n"
            "    if not isinstance(text, str):\n"
            "        return None\n"
            "    output = text\n"
            f"{replacement_lines}\n"
            "    return output\n"
        )
        omitted_program = (
            f"def {entrypoint}(text):\n"
            "    if not isinstance(text, str):\n"
            "        return None\n"
            "    output = text\n"
            f"{mutant_lines}\n"
            "    return output\n"
        )
        partial_control_program = omitted_program
        partial_control_defect = f"does not escape {omitted!r}"
        if entrypoint in {"escape_html_text", "escape_html_title"}:
            partial_control_program = (
                f"def {entrypoint}(text):\n"
                "    if not isinstance(text, str):\n"
                "        return None\n"
                "    return text.replace('&lt;', '&amp;lt;').replace('<', '&lt;').replace('>', '&gt;')\n"
            )
            partial_control_defect = "escapes ampersands only inside an existing lt entity"
        probe = f"safe{omitted}unsafe"
        expected_probe = probe
        for character, escaped in replacements.items():
            expected_probe = expected_probe.replace(character, escaped)
        rows.append(_task(
            task_id=f"cwe79-{entrypoint.replace('_', '-')}",
            cwe_id="CWE-79",
            cwe_name="Improper Neutralization of Input During Web Page Generation",
            split=Split.DEVELOPMENT,
            entrypoint=entrypoint,
            signature=f"{entrypoint}(text: str) -> str | None",
            spec=(
                "Return null for non-string input. For strings, replace characters in this exact order and "
                f"mapping: {list(replacements.items())}. Return the escaped string without adding markup."
            ),
            vulnerable=f"""
            def {entrypoint}(text):
                if not isinstance(text, str):
                    return None
                return text.replace("<", "&lt;").replace(">", "&gt;")
            """,
            reference=reference_program,
            cases=[
                _case("plain", ["hello"], "hello"),
                _case("ampersand-order", ["&lt;"], "&amp;lt;"),
                _case("context-char", [probe], expected_probe),
                _case("all", ["<&>\"'"], _escape_expected("<&>\"'", replacements)),
                _case("wrong-type", [7], None),
            ],
            mutants=[
                _mutant(f"{entrypoint}-omitted", partial_control_defect, partial_control_program),
                _mutant(f"{entrypoint}-amp-last", "escapes ampersand after introduced entities", f"""
                def {entrypoint}(text):
                    if not isinstance(text, str):
                        return None
                    output = text.replace("<", "&lt;").replace(">", "&gt;")
                    return output.replace("&", "&amp;")
                """),
            ],
        ))

    # Four parameterized-query CWE-89 variants form the fourth locked family.
    for entrypoint, table, column in (
        ("lookup_user", "users", "username"),
        ("lookup_order", "orders", "order_code"),
        ("lookup_device", "devices", "serial"),
        ("lookup_ticket", "tickets", "ticket_key"),
    ):
        sql = f"SELECT id FROM {table} WHERE {column} = ?"
        rows.append(_task(
            task_id=f"cwe89-{entrypoint.replace('_', '-')}",
            cwe_id="CWE-89",
            cwe_name="SQL Injection",
            split=Split.LOCKED_TEST,
            entrypoint=entrypoint,
            signature=f"{entrypoint}(value: str) -> list[object] | None",
            spec=(
                f"For a non-empty string, return the two-element JSON list [{sql!r}, [value]], preserving "
                "the value only as a bound parameter. Return null for other inputs or an empty string."
            ),
            vulnerable=f"""
            def {entrypoint}(value):
                if not isinstance(value, str) or not value:
                    return None
                return ["SELECT id FROM {table} WHERE {column} = '" + value + "'", []]
            """,
            reference=f"""
            def {entrypoint}(value):
                if not isinstance(value, str) or not value:
                    return None
                return [{sql!r}, [value]]
            """,
            cases=[
                _case("ordinary", ["alpha"], [sql, ["alpha"]]),
                _case("quote", ["O'Reilly"], [sql, ["O'Reilly"]]),
                _case("metacharacters", ["x' OR '1'='1"], [sql, ["x' OR '1'='1"]]),
                _case("empty", [""], None),
                _case("wrong-type", [7], None),
            ],
            mutants=[
                _mutant(f"{entrypoint}-interpolate", "interpolates apostrophes unless they resemble an obvious injection", f"""
                def {entrypoint}(value):
                    if not isinstance(value, str) or not value:
                        return None
                    if "'" in value and " OR " not in value:
                        return ["SELECT id FROM {table} WHERE {column} = '" + value.replace("'", "''") + "'", []]
                    return [{sql!r}, [value]]
                """),
                _mutant(f"{entrypoint}-wrong-bind", "uses a placeholder but omits the bound value", f"""
                def {entrypoint}(value):
                    if not isinstance(value, str) or not value:
                        return None
                    return [{sql!r}, []]
                """),
            ],
        ))
    return rows


def _escape_expected(text: str, replacements: dict[str, str]) -> str:
    output = text
    for character, escaped in replacements.items():
        output = output.replace(character, escaped)
    return output


@dataclass(frozen=True)
class CorpusBundle:
    public_tasks: tuple[PublicTask, ...]
    private_oracles: tuple[PrivateOracle, ...]

    def __post_init__(self) -> None:
        task_ids = [task.task_id for task in self.public_tasks]
        oracle_ids = [oracle.task_id for oracle in self.private_oracles]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("public task ids must be unique")
        if len(oracle_ids) != len(set(oracle_ids)):
            raise ValueError("private oracle task ids must be unique")
        if set(task_ids) != set(oracle_ids):
            raise ValueError("public tasks and private oracles must match one-to-one")
        family_splits: dict[str, set[Split]] = {}
        for task in self.public_tasks:
            family_splits.setdefault(task.cwe_id, set()).add(task.split)
        leaked = {family: splits for family, splits in family_splits.items() if len(splits) != 1}
        if leaked:
            raise ValueError(f"CWE families cross the development/test boundary: {leaked}")
        if len(family_splits) < 6:
            raise ValueError("the G0 corpus requires at least six CWE families")
        observed = {next(iter(splits)) for splits in family_splits.values()}
        if observed != {Split.DEVELOPMENT, Split.LOCKED_TEST}:
            raise ValueError("the corpus requires development and locked-test CWE families")

    @property
    def corpus_sha256(self) -> str:
        return stable_hash(
            {
                "schema_version": SCHEMA_VERSION,
                "public": [task.to_record() for task in self.public_tasks],
                "private": [oracle.to_record() for oracle in self.private_oracles],
            }
        )

    def task(self, task_id: str) -> PublicTask:
        return next(task for task in self.public_tasks if task.task_id == task_id)

    def oracle(self, task_id: str) -> PrivateOracle:
        return next(oracle for oracle in self.private_oracles if oracle.task_id == task_id)


def build_corpus() -> CorpusBundle:
    """Build the frozen 32-task, eight-family G0 corpus."""

    rows: list[tuple[PublicTask, PrivateOracle]] = []

    rows.append(_task(
        task_id="cwe20-age-ascii",
        cwe_id="CWE-20",
        cwe_name="Improper Input Validation",
        split=Split.DEVELOPMENT,
        entrypoint="parse_age",
        signature="parse_age(text: str) -> int | None",
        spec=(
            "Return the integer age for a non-empty string containing only ASCII digits "
            "when its value is between 0 and 130 inclusive. Return null for every other input."
        ),
        vulnerable="""
        def parse_age(text):
            if not isinstance(text, str):
                return None
            try:
                value = int(text.strip())
            except Exception:
                return None
            return value if 0 <= value <= 130 else None
        """,
        reference="""
        def parse_age(text):
            if not isinstance(text, str) or not text:
                return None
            if not all("0" <= char <= "9" for char in text):
                return None
            value = int(text)
            return value if 0 <= value <= 130 else None
        """,
        cases=[
            _case("zero", ["0"], 0),
            _case("upper", ["130"], 130),
            _case("above", ["131"], None),
            _case("leading-space", [" 42"], None),
            _case("plus", ["+7"], None),
            _case("unicode-digits", ["１２"], None),
            _case("wrong-type", [42], None),
        ],
        mutants=[
            _mutant("age-strip", "accepts surrounding whitespace", """
            def parse_age(text):
                if not isinstance(text, str):
                    return None
                text = text.strip()
                if not text or not text.isdigit():
                    return None
                value = int(text)
                return value if 0 <= value <= 130 else None
            """),
            _mutant("age-upper-open", "accepts age 131", """
            def parse_age(text):
                if not isinstance(text, str) or not text or not all("0" <= c <= "9" for c in text):
                    return None
                value = int(text)
                return value if 0 <= value <= 131 else None
            """),
        ],
    ))

    rows.append(_task(
        task_id="cwe20-port-ascii",
        cwe_id="CWE-20",
        cwe_name="Improper Input Validation",
        split=Split.DEVELOPMENT,
        entrypoint="parse_port",
        signature="parse_port(text: str) -> int | None",
        spec=(
            "Return a TCP port for a non-empty ASCII-decimal string in 1..65535. "
            "Whitespace, signs, non-ASCII digits, other types, zero, and out-of-range values return null."
        ),
        vulnerable="""
        def parse_port(text):
            if not isinstance(text, str) or not text.isdigit():
                return None
            value = int(text)
            return value if 0 <= value <= 65535 else None
        """,
        reference="""
        def parse_port(text):
            if not isinstance(text, str) or not text:
                return None
            if not all("0" <= char <= "9" for char in text):
                return None
            value = int(text)
            return value if 1 <= value <= 65535 else None
        """,
        cases=[
            _case("minimum", ["1"], 1),
            _case("maximum", ["65535"], 65535),
            _case("zero", ["0"], None),
            _case("above", ["65536"], None),
            _case("negative", ["-1"], None),
            _case("unicode", ["９０"], None),
        ],
        mutants=[
            _mutant("port-zero", "accepts reserved port zero", """
            def parse_port(text):
                if not isinstance(text, str) or not text or not all("0" <= c <= "9" for c in text):
                    return None
                value = int(text)
                return value if 0 <= value <= 65535 else None
            """),
            _mutant("port-unicode", "uses Unicode digit classification", """
            def parse_port(text):
                if not isinstance(text, str) or not text or not text.isdigit():
                    return None
                value = int(text)
                return value if 1 <= value <= 65535 else None
            """),
        ],
    ))

    rows.append(_task(
        task_id="cwe22-relative-path",
        cwe_id="CWE-22",
        cwe_name="Path Traversal",
        split=Split.DEVELOPMENT,
        entrypoint="safe_relative_path",
        signature="safe_relative_path(path: str) -> str | None",
        spec=(
            "Normalize a non-empty relative resource path to forward slashes. Collapse empty and '.' "
            "segments. Reject absolute paths, drive/URI colons, NUL, backslashes, '..' segments, and paths "
            "that normalize to empty. Return null on rejection."
        ),
        vulnerable="""
        def safe_relative_path(path):
            if not isinstance(path, str) or not path or path.startswith("/") or ".." in path:
                return None
            return path
        """,
        reference="""
        def safe_relative_path(path):
            if not isinstance(path, str) or not path or "\\x00" in path or "\\\\" in path:
                return None
            if path.startswith("/") or ":" in path:
                return None
            output = []
            for segment in path.split("/"):
                if segment in ("", "."):
                    continue
                if segment == "..":
                    return None
                output.append(segment)
            return "/".join(output) if output else None
        """,
        cases=[
            _case("simple", ["images/cat.png"], "images/cat.png"),
            _case("collapse", ["a//./b"], "a/b"),
            _case("parent", ["a/../secret"], None),
            _case("absolute", ["/etc/config"], None),
            _case("backslash", ["a\\..\\secret"], None),
            _case("drive", ["C:/temp/file"], None),
        ],
        mutants=[
            _mutant("path-colon", "fails to reject a drive or URI colon", """
            def safe_relative_path(path):
                if not isinstance(path, str) or not path or "\\x00" in path or "\\\\" in path:
                    return None
                if path.startswith("/"):
                    return None
                output = [part for part in path.split("/") if part not in ("", ".")]
                return None if ".." in output or not output else "/".join(output)
            """),
            _mutant("path-parent-substring", "rejects only a literal ../ substring", """
            def safe_relative_path(path):
                if not isinstance(path, str) or not path or path.startswith("/") or "../" in path or ":" in path:
                    return None
                output = [part for part in path.split("/") if part not in ("", ".")]
                return "/".join(output) if output else None
            """),
        ],
    ))

    rows.append(_task(
        task_id="cwe22-root-join",
        cwe_id="CWE-22",
        cwe_name="Path Traversal",
        split=Split.DEVELOPMENT,
        entrypoint="join_under_root",
        signature="join_under_root(root: str, requested: str) -> str | None",
        spec=(
            "Join an absolute canonical root with a relative requested path. The root must start with one '/' "
            "and contain no backslash, NUL, '.', or '..' segment. The request follows the same relative-path "
            "rules and may not contain empty, '.', or '..' segments. Return null when either input is invalid."
        ),
        vulnerable="""
        def join_under_root(root, requested):
            if not isinstance(root, str) or not isinstance(requested, str):
                return None
            if not root.startswith("/") or requested.startswith("/") or ".." in requested:
                return None
            return root.rstrip("/") + "/" + requested
        """,
        reference="""
        def join_under_root(root, requested):
            if not isinstance(root, str) or not isinstance(requested, str):
                return None
            if not root.startswith("/") or root.startswith("//") or "\\\\" in root or "\\x00" in root:
                return None
            root_parts = root.split("/")[1:]
            if not root_parts or any(part in ("", ".", "..") for part in root_parts):
                return None
            if not requested or requested.startswith("/") or "\\\\" in requested or "\\x00" in requested or ":" in requested:
                return None
            parts = requested.split("/")
            if any(part in ("", ".", "..") for part in parts):
                return None
            return "/" + "/".join(root_parts + parts)
        """,
        cases=[
            _case("join", ["/srv/data", "images/a.png"], "/srv/data/images/a.png"),
            _case("request-parent", ["/srv/data", "../secret"], None),
            _case("request-backslash", ["/srv/data", "a\\b"], None),
            _case("request-empty-segment", ["/srv/data", "a//b"], None),
            _case("bad-root", ["srv/data", "a.txt"], None),
            _case("root-parent", ["/srv/../data", "a.txt"], None),
        ],
        mutants=[
            _mutant("join-root-unchecked", "validates only the requested suffix", """
            def join_under_root(root, requested):
                if not isinstance(root, str) or not isinstance(requested, str) or not root.startswith("/"):
                    return None
                if not requested or requested.startswith("/") or ".." in requested:
                    return None
                return root.rstrip("/") + "/" + requested
            """),
            _mutant("join-empty-collapse", "silently collapses empty path segments", """
            def join_under_root(root, requested):
                if not isinstance(root, str) or not isinstance(requested, str) or not root.startswith("/"):
                    return None
                root_parts = [part for part in root.split("/") if part]
                parts = [part for part in requested.split("/") if part]
                if not parts or any(part in (".", "..") for part in root_parts + parts):
                    return None
                return "/" + "/".join(root_parts + parts)
            """),
        ],
    ))

    rows.append(_task(
        task_id="cwe200-header-redaction",
        cwe_id="CWE-200",
        cwe_name="Exposure of Sensitive Information",
        split=Split.DEVELOPMENT,
        entrypoint="redact_headers",
        signature="redact_headers(headers: dict[str, str]) -> dict[str, str] | None",
        spec=(
            "Return a new header dictionary, preserving names and non-secret values. Replace values with "
            "'[REDACTED]' when a header name case-insensitively equals authorization, cookie, set-cookie, "
            "x-api-key, or proxy-authorization. Return null unless all keys and values are strings."
        ),
        vulnerable="""
        def redact_headers(headers):
            if not isinstance(headers, dict):
                return None
            return {key: ("[REDACTED]" if key in ("Authorization", "Cookie") else value) for key, value in headers.items()}
        """,
        reference="""
        def redact_headers(headers):
            if not isinstance(headers, dict):
                return None
            secrets = {"authorization", "cookie", "set-cookie", "x-api-key", "proxy-authorization"}
            output = {}
            for key, value in headers.items():
                if not isinstance(key, str) or not isinstance(value, str):
                    return None
                output[key] = "[REDACTED]" if key.lower() in secrets else value
            return output
        """,
        cases=[
            _case("ordinary", [{"Accept": "application/json"}], {"Accept": "application/json"}),
            _case("ordinary-second", [{"Host": "example", "Accept": "text/plain"}], {"Host": "example", "Accept": "text/plain"}),
            _case("auth-case", [{"AUTHORIZATION": "Bearer secret"}], {"AUTHORIZATION": "[REDACTED]"}),
            _case("api-key", [{"x-api-key": "secret"}], {"x-api-key": "[REDACTED]"}),
            _case("set-cookie", [{"Set-Cookie": "sid=secret"}], {"Set-Cookie": "[REDACTED]"}),
            _case("mixed", [{"Host": "example", "cookie": "s"}], {"Host": "example", "cookie": "[REDACTED]"}),
            _case("bad-value", [{"Cookie": 7}], None),
        ],
        mutants=[
            _mutant("headers-case", "matches secret names case-sensitively", """
            def redact_headers(headers):
                if not isinstance(headers, dict):
                    return None
                secrets = {"Authorization", "Cookie", "Set-Cookie", "X-Api-Key", "Proxy-Authorization"}
                if not all(isinstance(k, str) and isinstance(v, str) for k, v in headers.items()):
                    return None
                return {k: ("[REDACTED]" if k in secrets else v) for k, v in headers.items()}
            """),
            _mutant("headers-short-list", "omits API and proxy credentials", """
            def redact_headers(headers):
                if not isinstance(headers, dict):
                    return None
                if not all(isinstance(k, str) and isinstance(v, str) for k, v in headers.items()):
                    return None
                return {k: ("[REDACTED]" if k.lower() in {"authorization", "cookie"} else v) for k, v in headers.items()}
            """),
        ],
    ))

    rows.append(_task(
        task_id="cwe200-public-profile",
        cwe_id="CWE-200",
        cwe_name="Exposure of Sensitive Information",
        split=Split.DEVELOPMENT,
        entrypoint="public_profile",
        signature="public_profile(profile: dict[str, object]) -> dict[str, str] | None",
        spec=(
            "Return a new dictionary containing only display_name and bio entries that are present. Both "
            "values must be strings; any non-string allowed value makes the input invalid. Ignore every "
            "other field. Return null when profile is not a dictionary."
        ),
        vulnerable="""
        def public_profile(profile):
            if not isinstance(profile, dict):
                return None
            output = dict(profile)
            output.pop("password", None)
            output.pop("token", None)
            return output
        """,
        reference="""
        def public_profile(profile):
            if not isinstance(profile, dict):
                return None
            output = {}
            for key in ("display_name", "bio"):
                if key in profile:
                    if not isinstance(profile[key], str):
                        return None
                    output[key] = profile[key]
            return output
        """,
        cases=[
            _case("both", [{"display_name": "Ada", "bio": "Engineer"}], {"display_name": "Ada", "bio": "Engineer"}),
            _case("strip-email", [{"display_name": "Ada", "email": "private@example"}], {"display_name": "Ada"}),
            _case("strip-new-secret", [{"bio": "Hi", "refresh_token": "secret"}], {"bio": "Hi"}),
            _case("empty", [{}], {}),
            _case("bad-allowed-value", [{"display_name": 7}], None),
            _case("wrong-type", [["Ada"]], None),
        ],
        mutants=[
            _mutant("profile-denylist", "uses an incomplete denylist instead of an allowlist", """
            def public_profile(profile):
                if not isinstance(profile, dict):
                    return None
                return {k: v for k, v in profile.items() if k not in {"password", "token"}}
            """),
            _mutant("profile-email", "mistakenly exposes email", """
            def public_profile(profile):
                if not isinstance(profile, dict):
                    return None
                output = {}
                for key in ("display_name", "bio", "email"):
                    if key in profile:
                        if not isinstance(profile[key], str):
                            return None
                        output[key] = profile[key]
                return output
            """),
        ],
    ))

    rows.append(_task(
        task_id="cwe287-owner-access",
        cwe_id="CWE-287",
        cwe_name="Improper Authentication",
        split=Split.LOCKED_TEST,
        entrypoint="can_access",
        signature="can_access(authenticated: bool, user_id: str, owner_id: str, is_admin: bool) -> bool",
        spec=(
            "Grant access only when authenticated and either is_admin is exactly true or the non-empty "
            "user_id exactly equals the non-empty owner_id. Malformed types must be denied."
        ),
        vulnerable="""
        def can_access(authenticated, user_id, owner_id, is_admin):
            return bool(is_admin or user_id == owner_id)
        """,
        reference="""
        def can_access(authenticated, user_id, owner_id, is_admin):
            if not isinstance(authenticated, bool) or not isinstance(is_admin, bool):
                return False
            if not isinstance(user_id, str) or not isinstance(owner_id, str):
                return False
            if not authenticated:
                return False
            return is_admin or (bool(user_id) and bool(owner_id) and user_id == owner_id)
        """,
        cases=[
            _case("owner", [True, "u1", "u1", False], True),
            _case("admin", [True, "u1", "u2", True], True),
            _case("unauth-owner", [False, "u1", "u1", False], False),
            _case("unauth-admin", [False, "u1", "u2", True], False),
            _case("empty-ids", [True, "", "", False], False),
            _case("truthy-admin", [True, "u1", "u2", 1], False),
        ],
        mutants=[
            _mutant("access-owner-before-auth", "owner equality bypasses authentication", """
            def can_access(authenticated, user_id, owner_id, is_admin):
                if not isinstance(authenticated, bool) or not isinstance(is_admin, bool):
                    return False
                return bool((authenticated and is_admin) or user_id == owner_id)
            """),
            _mutant("access-admin-bypass", "administrator flag bypasses authentication", """
            def can_access(authenticated, user_id, owner_id, is_admin):
                if not isinstance(authenticated, bool) or not isinstance(is_admin, bool):
                    return False
                return bool(is_admin or (authenticated and user_id and user_id == owner_id))
            """),
        ],
    ))

    rows.append(_task(
        task_id="cwe287-session-subject",
        cwe_id="CWE-287",
        cwe_name="Improper Authentication",
        split=Split.LOCKED_TEST,
        entrypoint="session_subject",
        signature="session_subject(session: dict[str, object], now: int) -> str | None",
        spec=(
            "Return user_id only for a dictionary session whose active field is exactly true, user_id is a "
            "non-empty string, expires_at is an integer (not bool), now is an integer (not bool), and "
            "expires_at is strictly greater than now. Otherwise return null."
        ),
        vulnerable="""
        def session_subject(session, now):
            if not isinstance(session, dict):
                return None
            if session.get("expires_at", 0) >= now:
                return session.get("user_id")
            return None
        """,
        reference="""
        def session_subject(session, now):
            if not isinstance(session, dict) or not isinstance(now, int) or isinstance(now, bool):
                return None
            if session.get("active") is not True:
                return None
            user_id = session.get("user_id")
            expires_at = session.get("expires_at")
            if not isinstance(user_id, str) or not user_id:
                return None
            if not isinstance(expires_at, int) or isinstance(expires_at, bool):
                return None
            return user_id if expires_at > now else None
        """,
        cases=[
            _case("valid", [{"active": True, "user_id": "u1", "expires_at": 101}, 100], "u1"),
            _case("valid-second", [{"active": True, "user_id": "u2", "expires_at": 250}, 200], "u2"),
            _case("expired-boundary", [{"active": True, "user_id": "u1", "expires_at": 100}, 100], None),
            _case("inactive", [{"active": False, "user_id": "u1", "expires_at": 101}, 100], None),
            _case("missing-active", [{"user_id": "u1", "expires_at": 101}, 100], None),
            _case("empty-user", [{"active": True, "user_id": "", "expires_at": 101}, 100], None),
            _case("bool-expiry", [{"active": True, "user_id": "u1", "expires_at": True}, 0], None),
        ],
        mutants=[
            _mutant("session-inclusive-expiry", "accepts a session at its expiry instant", """
            def session_subject(session, now):
                if not isinstance(session, dict) or session.get("active") is not True:
                    return None
                user_id = session.get("user_id")
                expiry = session.get("expires_at")
                if not isinstance(user_id, str) or not user_id or not isinstance(expiry, int):
                    return None
                return user_id if expiry >= now else None
            """),
            _mutant("session-active-omitted", "does not require an active session", """
            def session_subject(session, now):
                if not isinstance(session, dict):
                    return None
                user_id = session.get("user_id")
                expiry = session.get("expires_at")
                if not isinstance(user_id, str) or not user_id or not isinstance(expiry, int):
                    return None
                return user_id if expiry > now else None
            """),
        ],
    ))

    rows.append(_task(
        task_id="cwe400-batch-budget",
        cwe_id="CWE-400",
        cwe_name="Uncontrolled Resource Consumption",
        split=Split.LOCKED_TEST,
        entrypoint="accept_batch",
        signature="accept_batch(items: list[str], max_items: int, max_item_chars: int) -> bool",
        spec=(
            "Accept only a list of strings when max_items and max_item_chars are positive integers (not bool), "
            "the list has at most max_items entries, and every string has at most max_item_chars characters."
        ),
        vulnerable="""
        def accept_batch(items, max_items, max_item_chars):
            return isinstance(items, list) and len(items) <= max_items
        """,
        reference="""
        def accept_batch(items, max_items, max_item_chars):
            if not isinstance(items, list):
                return False
            if not isinstance(max_items, int) or isinstance(max_items, bool) or max_items <= 0:
                return False
            if not isinstance(max_item_chars, int) or isinstance(max_item_chars, bool) or max_item_chars <= 0:
                return False
            if len(items) > max_items:
                return False
            return all(isinstance(item, str) and len(item) <= max_item_chars for item in items)
        """,
        cases=[
            _case("valid", [["aa", "bbb"], 2, 3], True),
            _case("too-many", [["a", "b", "c"], 2, 3], False),
            _case("long-item", [["abcd"], 2, 3], False),
            _case("wrong-item", [["a", 2], 2, 3], False),
            _case("zero-limit", [[], 0, 3], False),
            _case("bool-limit", [["a"], True, 3], False),
        ],
        mutants=[
            _mutant("batch-count-only", "does not bound individual item size", """
            def accept_batch(items, max_items, max_item_chars):
                if not isinstance(items, list) or not isinstance(max_items, int) or not isinstance(max_item_chars, int):
                    return False
                return max_items > 0 and max_item_chars > 0 and len(items) <= max_items
            """),
            _mutant("batch-bool-limits", "accepts booleans as integer limits", """
            def accept_batch(items, max_items, max_item_chars):
                if not isinstance(items, list) or not isinstance(max_items, int) or not isinstance(max_item_chars, int):
                    return False
                if max_items <= 0 or max_item_chars <= 0 or len(items) > max_items:
                    return False
                return all(isinstance(item, str) and len(item) <= max_item_chars for item in items)
            """),
        ],
    ))

    rows.append(_task(
        task_id="cwe400-run-expansion",
        cwe_id="CWE-400",
        cwe_name="Uncontrolled Resource Consumption",
        split=Split.LOCKED_TEST,
        entrypoint="expand_runs",
        signature="expand_runs(runs: list[list[object]], max_output: int) -> list[object] | None",
        spec=(
            "Expand [value, count] pairs only when max_output is a positive integer (not bool), every count "
            "is a non-negative integer (not bool), and the total count is at most max_output. Validate the "
            "complete input and total before constructing output; otherwise return null."
        ),
        vulnerable="""
        def expand_runs(runs, max_output):
            output = []
            for value, count in runs:
                output.extend([value] * count)
            return output if len(output) <= max_output else None
        """,
        reference="""
        def expand_runs(runs, max_output):
            if not isinstance(runs, list) or not isinstance(max_output, int) or isinstance(max_output, bool) or max_output <= 0:
                return None
            total = 0
            for run in runs:
                if not isinstance(run, list) or len(run) != 2:
                    return None
                count = run[1]
                if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                    return None
                total += count
                if total > max_output:
                    return None
            output = []
            for value, count in runs:
                output.extend([value] * count)
            return output
        """,
        cases=[
            _case("valid", [[['a', 2], ['b', 1]], 3], ["a", "a", "b"]),
            _case("boundary", [[[1, 2], [2, 2]], 4], [1, 1, 2, 2]),
            _case("over-budget", [[["x", 3]], 2], None),
            _case("negative", [[["x", -1]], 2], None),
            _case("bool-count", [[["x", True]], 2], None),
            _case("bad-run", [[["x", 1, 2]], 3], None),
        ],
        mutants=[
            _mutant("runs-post-check", "allocates before enforcing the output budget", """
            def expand_runs(runs, max_output):
                if not isinstance(runs, list) or not isinstance(max_output, int) or max_output <= 0:
                    return None
                output = []
                for run in runs:
                    if not isinstance(run, list) or len(run) != 2 or not isinstance(run[1], int) or run[1] < 0:
                        return None
                    output.extend([run[0]] * run[1])
                return output if len(output) <= max_output else None
            """),
            _mutant("runs-strict-budget", "rejects output exactly at the budget", """
            def expand_runs(runs, max_output):
                if not isinstance(runs, list) or not isinstance(max_output, int) or isinstance(max_output, bool) or max_output <= 0:
                    return None
                total = 0
                for run in runs:
                    if not isinstance(run, list) or len(run) != 2 or not isinstance(run[1], int) or isinstance(run[1], bool) or run[1] < 0:
                        return None
                    total += run[1]
                if total >= max_output:
                    return None
                output = []
                for value, count in runs:
                    output.extend([value] * count)
                return output
            """),
        ],
    ))

    rows.append(_task(
        task_id="cwe601-local-next",
        cwe_id="CWE-601",
        cwe_name="URL Redirection to Untrusted Site",
        split=Split.LOCKED_TEST,
        entrypoint="safe_next_path",
        signature="safe_next_path(target: str) -> str | None",
        spec=(
            "Accept only a local absolute path beginning with exactly one '/'. Reject backslashes, NUL, "
            "control/space characters, colons, and '.' or '..' path segments. Preserve query and fragment "
            "suffixes after validating only the path portion. Return the unchanged target or null."
        ),
        vulnerable="""
        def safe_next_path(target):
            if isinstance(target, str) and target.startswith("/"):
                return target
            return None
        """,
        reference="""
        def safe_next_path(target):
            if not isinstance(target, str) or not target.startswith("/") or target.startswith("//"):
                return None
            if "\\\\" in target or "\\x00" in target or ":" in target:
                return None
            if any(ord(char) <= 32 for char in target):
                return None
            path = target.split("?", 1)[0].split("#", 1)[0]
            if any(part in (".", "..") for part in path.split("/")):
                return None
            return target
        """,
        cases=[
            _case("local", ["/account/settings"], "/account/settings"),
            _case("query", ["/search?q=cat"], "/search?q=cat"),
            _case("scheme-relative", ["//evil.example/x"], None),
            _case("backslash", ["/\\evil.example/x"], None),
            _case("parent", ["/safe/../admin"], None),
            _case("colon", ["/https:evil"], None),
        ],
        mutants=[
            _mutant("next-prefix-only", "accepts scheme-relative redirects", """
            def safe_next_path(target):
                if not isinstance(target, str) or not target.startswith("/"):
                    return None
                if "\\\\" in target or "\\x00" in target:
                    return None
                return target
            """),
            _mutant("next-no-traversal", "does not reject parent path segments", """
            def safe_next_path(target):
                if not isinstance(target, str) or not target.startswith("/") or target.startswith("//"):
                    return None
                if "\\\\" in target or "\\x00" in target or ":" in target:
                    return None
                return target
            """),
        ],
    ))

    rows.append(_task(
        task_id="cwe601-https-host",
        cwe_id="CWE-601",
        cwe_name="URL Redirection to Untrusted Site",
        split=Split.LOCKED_TEST,
        entrypoint="allow_https_redirect",
        signature="allow_https_redirect(target: str, allowed_hosts: list[str]) -> str | None",
        spec=(
            "Accept an https URL only when its authority is a lowercase-insensitive exact match for one of "
            "the non-empty allowed host strings. Reject credentials, ports, backslashes, whitespace/control "
            "characters, fragments before authority parsing, malformed host lists, and every other scheme."
        ),
        vulnerable="""
        def allow_https_redirect(target, allowed_hosts):
            if not isinstance(target, str) or not target.startswith("https://"):
                return None
            return target if any(host in target for host in allowed_hosts) else None
        """,
        reference="""
        def allow_https_redirect(target, allowed_hosts):
            if not isinstance(target, str) or not isinstance(allowed_hosts, list):
                return None
            if not target.lower().startswith("https://") or "\\\\" in target:
                return None
            if any(ord(char) <= 32 for char in target):
                return None
            if not all(isinstance(host, str) and host and host == host.strip() and "/" not in host and "@" not in host and ":" not in host for host in allowed_hosts):
                return None
            remainder = target[8:]
            authority = remainder.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
            if not authority or "@" in authority or ":" in authority:
                return None
            allowed = {host.lower() for host in allowed_hosts}
            return target if authority.lower() in allowed else None
        """,
        cases=[
            _case("allowed", ["https://example.com/account", ["example.com"]], "https://example.com/account"),
            _case("case-host", ["HTTPS://EXAMPLE.COM/x", ["example.com"]], "HTTPS://EXAMPLE.COM/x"),
            _case("suffix-host", ["https://example.com.evil.test/x", ["example.com"]], None),
            _case("userinfo", ["https://example.com@evil.test/x", ["example.com"]], None),
            _case("port", ["https://example.com:443/x", ["example.com"]], None),
            _case("http", ["http://example.com/x", ["example.com"]], None),
        ],
        mutants=[
            _mutant("redirect-substring", "uses host substring matching", """
            def allow_https_redirect(target, allowed_hosts):
                if not isinstance(target, str) or not isinstance(allowed_hosts, list) or not target.lower().startswith("https://"):
                    return None
                return target if any(host.lower() in target.lower() for host in allowed_hosts) else None
            """),
            _mutant("redirect-userinfo", "extracts the prefix before @ as the host", """
            def allow_https_redirect(target, allowed_hosts):
                if not isinstance(target, str) or not isinstance(allowed_hosts, list) or not target.lower().startswith("https://"):
                    return None
                authority = target[8:].split("/", 1)[0].split("@", 1)[0]
                return target if authority.lower() in {host.lower() for host in allowed_hosts} else None
            """),
        ],
    ))

    rows.extend(_additional_tasks())
    rows.sort(key=lambda pair: pair[0].task_id)
    return CorpusBundle(
        public_tasks=tuple(public for public, _ in rows),
        private_oracles=tuple(private for _, private in rows),
    )


def _jsonl_bytes(records: list[dict[str, object]]) -> bytes:
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    ).encode("utf-8")


def write_corpus(destination: Path) -> dict[str, object]:
    """Write public/private artifacts exactly once and return their manifest.

    ``destination`` must not exist.  Every file uses exclusive creation and the
    manifest contains no timestamps, so identical corpus versions hash equally.
    """

    bundle = build_corpus()
    public_bytes = _jsonl_bytes([task.to_record() for task in bundle.public_tasks])
    private_bytes = _jsonl_bytes([oracle.to_record() for oracle in bundle.private_oracles])
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "validator_monoculture_security_repair_g0",
        "task_count": len(bundle.public_tasks),
        "cwe_families": sorted({task.cwe_id for task in bundle.public_tasks}),
        "development_cwe_families": sorted({
            task.cwe_id for task in bundle.public_tasks if task.split is Split.DEVELOPMENT
        }),
        "locked_test_cwe_families": sorted({
            task.cwe_id for task in bundle.public_tasks if task.split is Split.LOCKED_TEST
        }),
        "corpus_sha256": bundle.corpus_sha256,
        "public_tasks_sha256": hashlib.sha256(public_bytes).hexdigest(),
        "private_oracles_sha256": hashlib.sha256(private_bytes).hexdigest(),
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")

    destination.mkdir(parents=True, exist_ok=False)
    public_dir = destination / "public"
    private_dir = destination / "private"
    public_dir.mkdir(exist_ok=False)
    private_dir.mkdir(exist_ok=False)
    for path, payload in (
        (public_dir / "tasks.jsonl", public_bytes),
        (private_dir / "oracles.jsonl", private_bytes),
        (destination / "MANIFEST.json", manifest_bytes),
    ):
        with path.open("xb") as handle:
            handle.write(payload)
    return manifest
