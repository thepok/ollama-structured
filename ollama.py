"""Utilities for calling Ollama and instantiating structured results."""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import json
import logging
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
    return "\n".join(lines)


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
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        candidate = _extract_json_substring(stripped)
        if candidate is None:
            raise OllamaStructuredError("Response did not contain valid JSON content") from None
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise OllamaStructuredError(
                f"Response did not contain valid JSON content: {exc}"
            ) from exc


def _extract_json_substring(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _instantiate_class(cls: type[Any], payload: Any) -> Any:
    if not isinstance(payload, dict):
        raise OllamaStructuredError(
            f"JSON payload must be an object to instantiate {cls.__qualname__}, got {type(payload).__name__}."
        )
    try:
        if _PydanticBaseModel is not None and issubclass(cls, _PydanticBaseModel):
            if hasattr(cls, "model_validate"):
                return cls.model_validate(payload)
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
