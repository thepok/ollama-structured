"""Utilities for calling Ollama and instantiating structured results."""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable, TypeVar, get_args, get_origin, get_type_hints

import requests

try:  # Optional dependency for richer validation.
    from pydantic import BaseModel as _PydanticBaseModel  # type: ignore
except Exception:  # pragma: no cover - pydantic not installed
    _PydanticBaseModel = None  # type: ignore


T = TypeVar("T")


class OllamaStructuredError(RuntimeError):
    """Raised when structured Ollama runs fail."""


@dataclasses.dataclass
class _FieldSpec:
    name: str
    type_repr: str
    required: bool
    description: str | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_logger() -> logging.Logger:
    logs_dir = _repo_root() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ollama_structured")
    if not logger.handlers:
        handler = logging.FileHandler(logs_dir / "ollama_structured.log", encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


class OllamaClient:
    """Call Ollama with schema instructions and materialize Python objects."""

    def __init__(
        self,
        *,
        model: str = "gpt-oss:20b",
        endpoint: str = "http://localhost:11434/api/generate",
        timeout_seconds: float = 3600.0,
        include_thinking: bool = True,
        logger: logging.Logger | None = None,
    ) -> None:
        self._model = model
        self._endpoint = endpoint
        self._timeout_seconds = timeout_seconds
        self._include_thinking = include_thinking
        self._logger = logger or _default_logger()

    def run(self, prompt: str, class_def: type[T] | str) -> T:
        """Execute ``prompt`` and return an instance of ``class_def`` filled with model output."""

        resolved_class = _resolve_base_class(class_def)
        field_specs, docstring = _introspect_class_schema(resolved_class)
        if not field_specs:
            raise OllamaStructuredError(
                f"Base class {resolved_class.__qualname__} has no discoverable fields for JSON mode."
            )

        schema_text = _render_field_specs(resolved_class, field_specs, docstring)
        structured_prompt = _build_json_prompt(prompt, resolved_class, schema_text)
        self._logger.info("Structured prompt:\n%s", structured_prompt)
        raw_response = self._call_ollama(structured_prompt)
        self._logger.info("Structured output (raw JSON text): %s", raw_response)
        payload = _load_json_payload(raw_response)
        instance = _instantiate_class(resolved_class, payload)
        return instance

    def _call_ollama(self, prompt: str) -> str:
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {"include_thinking": self._include_thinking},
        }
        payload_json = json.dumps(payload, ensure_ascii=True)
        self._logger.info("Sending Ollama request: %s", payload_json)

        try:
            response = requests.post(
                self._endpoint,
                json=payload,
                timeout=self._timeout_seconds,
            )
        except requests.RequestException as exc:  # pragma: no cover
            self._logger.exception("HTTP request to Ollama failed")
            raise OllamaStructuredError("Failed to contact Ollama endpoint") from exc

        raw_output = (response.text or "").strip()
        self._logger.info(
            "Ollama HTTP status: %s %s", response.status_code, response.reason
        )
        self._logger.info("Raw Ollama response: %s", raw_output)

        if response.status_code != 200:
            raise OllamaStructuredError(
                f"Ollama returned HTTP {response.status_code} ({response.reason})"
            )

        if not raw_output:
            raise OllamaStructuredError("Ollama returned an empty response body")

        try:
            parsed = response.json()
        except ValueError as exc:
            self._logger.exception("Failed to parse Ollama JSON payload")
            raise OllamaStructuredError("Ollama response was not valid JSON") from exc

        thinking, final_output = _extract_reasoning(parsed)
        if thinking:
            self._logger.debug("Model thinking: %s", thinking)
        if not final_output:
            raise OllamaStructuredError("Ollama response did not contain final output text")
        return final_output


def _extract_reasoning(payload: Any) -> tuple[str, str]:
    """Return (thinking, final_output) from the Ollama JSON payload."""

    def _stringify(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return "".join(_stringify(item) for item in value)
        if isinstance(value, dict):
            for key in (
                "output",
                "result",
                "response",
                "final",
                "answer",
                "content",
                "value",
                "text",
            ):
                if key in value:
                    candidate = _stringify(value[key])
                    if candidate:
                        return candidate
            return "\n".join(_stringify(v) for v in value.values())
        return ""

    thinking = ""
    final_output = ""
    if isinstance(payload, dict):
        thinking = _stringify(payload.get("thinking"))
        response = payload.get("response")
        if response:
            final_output = _stringify(response)
        if not final_output:
            for key in ("result", "output", "final", "answer", "content"):
                if key in payload:
                    final_output = _stringify(payload[key])
                    if final_output:
                        break
    elif isinstance(payload, list):
        thoughts: list[str] = []
        outputs: list[str] = []
        for item in payload:
            t, o = _extract_reasoning(item)
            if t:
                thoughts.append(t)
            if o:
                outputs.append(o)
        thinking = "\n".join(thoughts)
        final_output = "".join(outputs)
    elif isinstance(payload, str):
        final_output = payload
    return thinking.strip(), final_output.strip()


def _resolve_base_class(base_class: type[Any] | str) -> type[Any]:
    if isinstance(base_class, type):
        return base_class
    if not isinstance(base_class, str):
        raise TypeError("base_class must be a type or import string")

    module_name: str
    class_name: str
    if ":" in base_class:
        module_name, class_name = base_class.split(":", 1)
    else:
        parts = base_class.rsplit(".", 1)
        if len(parts) != 2:
            raise ValueError(
                "base_class import string must be 'module:ClassName' or 'module.ClassName'."
            )
        module_name, class_name = parts

    module = importlib.import_module(module_name)
    try:
        resolved = getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(
            f"Class {class_name!r} not found in module {module_name!r}."
        ) from exc

    if not isinstance(resolved, type):
        raise TypeError(f"Resolved object {resolved!r} is not a class")
    return resolved


def _introspect_class_schema(
    cls: type[Any],
) -> tuple[list[_FieldSpec], str | None]:
    docstring = inspect.getdoc(cls)

    if dataclasses.is_dataclass(cls):
        type_hints = get_type_hints(cls)
        specs: list[_FieldSpec] = []
        for field in dataclasses.fields(cls):
            annotation = type_hints.get(field.name, field.type)
            description = None
            if field.metadata and "description" in field.metadata:
                description = str(field.metadata["description"])
            required = field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING
            specs.append(
                _FieldSpec(
                    name=field.name,
                    type_repr=_format_type(annotation),
                    required=required,
                    description=description,
                )
            )
        return specs, docstring

    if _PydanticBaseModel is not None and issubclass(cls, _PydanticBaseModel):
        specs = []
        model_fields = getattr(cls, "model_fields", None) or getattr(cls, "__fields__", None)
        for name, field in (model_fields or {}).items():
            annotation = getattr(field, "annotation", None) or getattr(field, "outer_type_", Any)
            description = getattr(field, "description", None)
            required = bool(getattr(field, "is_required", lambda: False)())
            if hasattr(field, "required"):
                required = bool(field.required)
            specs.append(
                _FieldSpec(
                    name=name,
                    type_repr=_format_type(annotation),
                    required=required,
                    description=description,
                )
            )
        return specs, docstring

    specs = []
    init = cls.__init__
    if not callable(init):
        return specs, docstring
    sig = inspect.signature(init)
    for name, param in sig.parameters.items():
        if name == "self" or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        specs.append(
            _FieldSpec(
                name=name,
                type_repr=_format_type(param.annotation),
                required=param.default is inspect._empty,
                description=None,
            )
        )
    return specs, docstring


def _format_type(annotation: Any) -> str:
    if annotation is inspect._empty or annotation is None:
        return "Any"
    if isinstance(annotation, type):
        return annotation.__name__
    origin = get_origin(annotation)
    if origin is None:
        return str(annotation)
    args = ", ".join(_format_type(arg) for arg in get_args(annotation))
    origin_name = getattr(origin, "__name__", str(origin))
    return f"{origin_name}[{args}]"


def _render_field_specs(
    cls: type[Any], fields: Iterable[_FieldSpec], docstring: str | None
) -> str:
    lines = [f"Base class: {cls.__module__}.{cls.__qualname__}"]
    if docstring:
        lines.append(f"Description: {docstring.strip()}")
    lines.append("Fields:")
    for field in fields:
        parts = [
            f"- {field.name}",
            f"type={field.type_repr}",
            "required" if field.required else "optional",
        ]
        if field.description:
            parts.append(field.description)
        lines.append("  " + " | ".join(parts))
    nested = _render_nested_dataclasses(cls)
    if nested:
        lines.append("")
        lines.append(nested)
    return "\n".join(lines)


def _render_nested_dataclasses(root_cls: type[Any]) -> str:
    """Describe dataclass fields reachable from the root dataclass.

    Some models ignore list element type hints unless we repeat nested schemas.
    """

    lines: list[str] = []
    seen: set[type[Any]] = set()

    def visit_type(tp: Any) -> None:
        origin = get_origin(tp)
        if origin is not None:
            for arg in get_args(tp):
                visit_type(arg)
            return
        if inspect.isclass(tp) and dataclasses.is_dataclass(tp):
            visit_class(tp)

    def visit_class(cls: type[Any]) -> None:
        if cls in seen:
            return
        seen.add(cls)

        type_hints = get_type_hints(cls)
        for field in dataclasses.fields(cls):
            visit_type(type_hints.get(field.name, field.type))

        if cls is root_cls:
            return

        lines.append(f"Nested class: {cls.__module__}.{cls.__qualname__}")
        lines.append("Fields:")
        for field in dataclasses.fields(cls):
            type_hints = get_type_hints(cls)
            ann = type_hints.get(field.name, field.type)
            required = field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING
            lines.append(
                "  "
                + " | ".join(
                    [
                        f"- {field.name}",
                        f"type={_format_type(ann)}",
                        "required" if required else "optional",
                    ]
                )
            )
        lines.append("")

    visit_class(root_cls)
    return "\n".join(line for line in lines if line).strip()


def _build_json_prompt(
    user_prompt: str, base_class: type[Any], schema_text: str
) -> str:
    requirements = "\n".join(
        [
            "Output requirements:",
            "- Respond with a single JSON object and nothing else.",
            "- Use double quotes for all keys and strings.",
            "- Populate every required field; omit optional fields only if unknown.",
            "- Use ISO 8601 timestamps ending with 'Z' when returning datetimes.",
        ]
    )
    return (
        f"{user_prompt.strip()}\n\n"
        f"{requirements}\n"
        f"\nSchema guidance (match {base_class.__qualname__}):\n{schema_text}"
    ).strip()


def _load_json_payload(raw_text: str) -> Any:
    stripped = raw_text.strip()
    if not stripped:
        raise OllamaStructuredError("Model response was empty; expected JSON content")

    # 1) Strict JSON first.
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as exc:
        last_exc: Exception = exc

    # 2) Remove common Markdown fences (```json ... ```).
    unfenced = _strip_code_fences(stripped)
    if unfenced != stripped:
        try:
            return json.loads(unfenced)
        except json.JSONDecodeError as exc:
            last_exc = exc

    # 3) Extract the first balanced JSON-like object/array from surrounding text.
    extracted = _extract_first_balanced_json(unfenced) or _extract_json_substring(unfenced)
    if extracted is None:
        raise OllamaStructuredError("Response did not contain valid JSON content") from None

    # 4) Try strict JSON again with a tiny sanitizer (trailing commas).
    sanitized = _remove_trailing_commas(extracted)
    for candidate in (extracted, sanitized):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_exc = exc

    # 5) Fallback: accept Python-literal dict/list (e.g., single quotes) via ast.literal_eval.
    # This is common when models "almost" output JSON. We still return normal Python
    # structures for downstream instantiation.
    try:
        import ast

        parsed = ast.literal_eval(extracted)
        return _normalize_literal_eval(parsed)
    except Exception as exc:
        raise OllamaStructuredError(
            f"Response did not contain valid JSON content: {last_exc}"
        ) from exc


def _extract_json_substring(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


_FENCE_RE = re.compile(r"^```(?:json|JSON)?\s*([\s\S]*?)\s*```$", re.MULTILINE)


def _strip_code_fences(text: str) -> str:
    match = _FENCE_RE.match(text.strip())
    if not match:
        return text.strip()
    return (match.group(1) or "").strip()


_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _remove_trailing_commas(text: str) -> str:
    # Repeatedly remove trailing commas before closing braces/brackets.
    prev = None
    cur = text
    for _ in range(10):
        prev = cur
        cur = _TRAILING_COMMA_RE.sub(r"\1", cur)
        if cur == prev:
            break
    return cur


def _extract_first_balanced_json(text: str) -> str | None:
    """Return the first balanced {...} or [...] substring, ignoring braces in strings."""
    s = text
    start = None
    stack: list[str] = []
    in_string = False
    escape = False

    def is_open(ch: str) -> bool:
        return ch in "{["

    def is_close(ch: str) -> bool:
        return ch in "}]"

    def matches(open_ch: str, close_ch: str) -> bool:
        return (open_ch == "{" and close_ch == "}") or (open_ch == "[" and close_ch == "]")

    for i, ch in enumerate(s):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue

        if start is None:
            if is_open(ch):
                start = i
                stack.append(ch)
            continue

        if is_open(ch):
            stack.append(ch)
            continue
        if is_close(ch) and stack and matches(stack[-1], ch):
            stack.pop()
            if not stack and start is not None:
                return s[start : i + 1]

    return None


def _normalize_literal_eval(value: Any) -> Any:
    """Normalize ast.literal_eval output into JSON-like structures."""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            # JSON requires string keys; most schemas expect that anyway.
            out[str(k)] = _normalize_literal_eval(v)
        return out
    if isinstance(value, (list, tuple, set)):
        return [_normalize_literal_eval(v) for v in value]
    return value


def _instantiate_class(cls: type[Any], payload: Any) -> Any:
    if not isinstance(payload, dict):
        raise OllamaStructuredError(
            f"JSON payload must be an object to instantiate {cls.__qualname__}, got {type(payload).__name__}."
        )

    def _convert_value(value: Any, hint: Any) -> Any:
        if value is None:
            return None

        origin = get_origin(hint)
        args = get_args(hint)

        # Optional[T] -> unwrap to T
        if origin is not None and type(None) in args:
            non_none = [a for a in args if a is not type(None)]
            hint = non_none[0] if len(non_none) == 1 else hint
            origin = get_origin(hint)
            args = get_args(hint)

        # Dataclass recursion
        if inspect.isclass(hint) and dataclasses.is_dataclass(hint):
            return _instantiate_class(hint, value)

        # Collections
        if origin in (list, tuple, set):
            elem_type = args[0] if args else Any
            converted = [_convert_value(v, elem_type) for v in (value or [])]
            if origin is tuple:
                return tuple(converted)
            if origin is set:
                return set(converted)
            return list(converted)

        if origin is dict and len(args) == 2:
            key_t, val_t = args
            return { _convert_value(k, key_t): _convert_value(v, val_t) for k, v in (value or {}).items() }

        return value

    try:
        if _PydanticBaseModel is not None and issubclass(cls, _PydanticBaseModel):
            if hasattr(cls, "model_validate"):
                return cls.model_validate(payload)

        if dataclasses.is_dataclass(cls):
            type_hints = get_type_hints(cls)
            kwargs = {}
            for field in dataclasses.fields(cls):
                hint = type_hints.get(field.name, field.type)
                if field.name in payload:
                    kwargs[field.name] = _convert_value(payload[field.name], hint)
            return cls(**kwargs)

        return cls(**payload)
    except TypeError as exc:
        raise OllamaStructuredError(
            f"Failed to instantiate {cls.__qualname__}: {exc}"
        ) from exc
    except ValueError as exc:
        raise OllamaStructuredError(
            f"Base class {cls.__qualname__} rejected payload: {exc}"
        ) from exc


__all__ = ["OllamaClient", "OllamaStructuredError"]
