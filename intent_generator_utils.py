"""Utility helpers for Intent Schema generation within the Workflow Builder app."""
from __future__ import annotations

import json
import os
import re
import subprocess
import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import networkx as nx
from pydantic import BaseModel, Field

JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


class McpArg(BaseModel):
    """Argument specification for an MCP tool."""

    name: str
    type: str
    required: bool = False
    default: Optional[Any] = None


class McpTool(BaseModel):
    """Metadata describing a callable MCP tool."""

    name: str
    description: str
    args: List[McpArg] = Field(default_factory=list)
    execution: Dict[str, Any]

    class Config:
        extra = "allow"


class IntentEntity(BaseModel):
    """Entity definition within an intent schema."""

    name: str
    type: Union[str, Dict[str, Any]]
    required: bool = False
    extract_rules: Optional[Union[str, List[Any]]] = None
    validate: Optional[Union[str, List[Any], Dict[str, Any]]] = None

    class Config:
        extra = "allow"


class IntentTaskNode(BaseModel):
    """Task graph node definition."""

    task_name: str
    description: str
    type: str
    tool: Optional[str] = None
    inputs: Optional[Union[List[Any], Dict[str, Any]]] = None
    condition: Optional[str] = None
    ui_hints: Optional[List[str]] = None

    class Config:
        extra = "allow"


class IntentSchema(BaseModel):
    """Top-level intent schema container."""

    name: str
    summary: str
    entities: List[IntentEntity]
    clarify: Dict[str, Any]
    task_graph: Dict[str, Any]

    class Config:
        extra = "allow"


INTENT_CONTRACT: Dict[str, Any] = {
    "type": "object",
    "required": ["name", "summary", "entities", "clarify", "task_graph"],
    "properties": {
        "name": {"type": "string"},
        "summary": {"type": "string"},
        "entities": {
            "type": "array",
            "items": {"type": "object", "required": ["name", "type"]},
        },
        "clarify": {"type": "object"},
        "task_graph": {
            "type": "object",
            "required": ["nodes"],
            "properties": {"nodes": {"type": "array"}},
        },
    },
}


def load_json_file(path: Union[str, Path]) -> Any:
    """Read JSON from disk and return the loaded object."""

    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_tools(path: Union[str, Path]) -> List[McpTool]:
    """Load tool specifications from disk."""

    data = load_json_file(path)
    if isinstance(data, dict):
        # Some catalogs are wrapped in an object with a key like "tools".
        if "tools" in data and isinstance(data["tools"], list):
            data = data["tools"]
        else:
            raise ValueError("tools.json must be a list or contain a 'tools' list")
    if not isinstance(data, list):
        raise ValueError("tools.json expected to be a list of tool objects")
    return [McpTool(**item) for item in data]


def load_intents(paths: Sequence[Union[str, Path]]) -> List[IntentSchema]:
    """Load multiple intent schema samples."""

    intents: List[IntentSchema] = []
    for path in paths:
        try:
            payload = load_json_file(path)
            intents.append(IntentSchema(**payload))
        except Exception as exc:  # pragma: no cover - surfaced in UI instead
            raise ValueError(f"Failed to load intent from {path}: {exc}")
    return intents


def minify_json(obj: Any) -> str:
    """Return a compact JSON string for prompt inclusion."""

    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def _workflow_text(workflow: Optional[Dict[str, Any]]) -> str:
    if not workflow:
        return ""
    nodes = workflow.get("nodes", [])
    edges = workflow.get("edges", [])
    parts = []
    for node in nodes:
        parts.append(str(node.get("label") or node.get("id") or ""))
        parts.append(str(node.get("type", "")))
    for edge in edges:
        parts.append(str(edge.get("label", "")))
    return " ".join(parts)


def _tokenize(text: str) -> List[str]:
    return [tok for tok in re.split(r"[^a-z0-9]+", text.lower()) if tok]


def choose_candidate_tools(
    workflow: Optional[Dict[str, Any]],
    manual_text: str,
    tools: Sequence[McpTool],
    top_k: int = 5,
) -> List[McpTool]:
    """Heuristically choose candidate tools based on workflow context."""

    context_text = " ".join([manual_text or "", _workflow_text(workflow)])
    context_tokens = set(_tokenize(context_text))
    if not context_tokens:
        return list(tools[: top_k or len(tools)])

    entity_hints = {tok for tok in context_tokens if len(tok) > 3}

    scored: List[Tuple[float, McpTool]] = []
    for tool in tools:
        haystack = " ".join(
            [
                tool.name,
                tool.description or "",
                " ".join(arg.name for arg in tool.args or []),
            ]
        ).lower()
        ratio = 0.0
        if haystack:
            matches = sum(1 for tok in context_tokens if tok in haystack)
            ratio = matches / max(len(context_tokens), 1)
        arg_bonus = 0.0
        for arg in tool.args or []:
            arg_name = arg.name.lower()
            if arg_name in context_tokens:
                arg_bonus += 0.2
            elif any(arg_name in hint for hint in entity_hints):
                arg_bonus += 0.1
        scored.append((ratio + arg_bonus, tool))

    scored.sort(key=lambda item: item[0], reverse=True)
    selected = [tool for _, tool in scored[:top_k]]

    # Always include explicitly referenced tool names.
    context_lower = context_text.lower()
    explicit = [tool for tool in tools if tool.name.lower() in context_lower]
    for tool in explicit:
        if tool not in selected:
            selected.append(tool)
    return selected[:top_k]


def summarize_workflow(workflow: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Create a concise summary of the workflow structure."""

    if not workflow:
        return {
            "direction": "unknown",
            "steps": [],
            "branches": [],
        }
    nodes = workflow.get("nodes", [])
    edges = workflow.get("edges", [])
    direction = workflow.get("direction", "TD")
    steps = [str(node.get("label") or node.get("id")) for node in nodes]

    graph = nx.DiGraph()
    for node in nodes:
        graph.add_node(node.get("id"), label=node.get("label"))
    for edge in edges:
        graph.add_edge(edge.get("source"), edge.get("target"), label=edge.get("label"))

    branches: List[str] = []
    for node_id in graph.nodes:
        succ = list(graph.successors(node_id))
        if len(succ) > 1:
            node_label = next(
                (n.get("label") for n in nodes if n.get("id") == node_id), node_id
            )
            branch_desc = ", ".join(
                f"{next((n.get('label') for n in nodes if n.get('id') == s), s)}"
                f" ({graph.get_edge_data(node_id, s).get('label', '').strip() or '→'})"
                for s in succ
            )
            branches.append(f"{node_label}: {branch_desc}")

    return {
        "direction": direction,
        "steps": steps,
        "branches": branches,
    }


def _tool_catalog_section(tools: Sequence[McpTool]) -> str:
    lines = ["TOOLS (subset):"]
    for tool in tools:
        args_repr = ", ".join(
            f"{arg.name}:{arg.type}{' (required)' if arg.required else ''}"
            + (f"={arg.default}" if arg.default not in (None, "") else "")
            for arg in tool.args or []
        )
        lines.append(f"- {tool.name}: {tool.description}")
        if args_repr:
            lines.append(f"  args: [{args_repr}]")
        else:
            lines.append("  args: []")
    return "\n".join(lines)


SYSTEM_PROMPT_BASE = """You are an expert Intent Schema author. Given:
1) A catalog of callable tools (name, arguments, purposes), and
2) 1–3 gold-standard sample intent schemas,
3) A user-authored workflow (graph summary) describing a new capability,

Produce a SINGLE JSON object that strictly follows our intent schema format. Your output will be validated and executed by our agentic framework.

Rules:
- Adhere EXACTLY to the schema fields and nesting used in the samples (names, casing, arrays, object shapes).
- Prefer tool names and required args as they appear in the tool catalog.
- Ensure every `task_graph.nodes[*].tool` maps to a real tool in the catalog.
- Every tool input must be satisfiable from `entities` or constants.
- Put entity extraction/validation rules under `entities[*].extract_rules` / `validate`.
- Capture clarification logic in `clarify` (rules, ask_when_missing, ui_hints, examples).
- Add `task_graph.guidelines` as concrete do’s/don’ts for the agent.
- If the workflow implies multiple intents, synthesize a single BEST intent that fulfills the end-to-end goal.
- Output ONLY the JSON object (no commentary), inside a fenced JSON block."""


def compose_prompt(
    workflow: Optional[Dict[str, Any]],
    manual_notes: str,
    tools_subset: Sequence[McpTool],
    sample_intents: Sequence[IntentSchema],
    contract: Dict[str, Any] = INTENT_CONTRACT,
) -> Tuple[str, str]:
    """Compose system and user prompts for the LLM."""

    tool_section = _tool_catalog_section(tools_subset)
    sample_blocks: List[str] = []
    for idx, sample in enumerate(sample_intents[:3], 1):
        sample_blocks.append(
            f"SAMPLE_INTENT_{idx} (minified):\n{minify_json(sample.dict())}"
        )
    samples_section = "\n\n".join(sample_blocks) if sample_blocks else ""
    contract_section = json.dumps(contract, indent=2)

    system_message = "\n\n".join(
        [part for part in [SYSTEM_PROMPT_BASE, tool_section, samples_section, f"STRICT OUTPUT CONTRACT:\n{contract_section}"] if part]
    )

    summary = summarize_workflow(workflow)
    steps = ", ".join(summary["steps"]) if summary["steps"] else "(none)"
    branches = "; ".join(summary["branches"]) if summary["branches"] else "(none)"

    user_message = (
        "WORKFLOW_CONTEXT:\n"
        f"- Direction: {summary['direction']}\n"
        f"- Steps: {steps}\n"
        f"- Branches: {branches}\n"
        f"- Notes: {manual_notes.strip() or 'None'}\n\n"
        "REQUIREMENTS:\n"
        "- Use the TOOLS subset above.\n"
        "- Copy sample structure faithfully; adapt names, entities, clarify rules, and task graph to this workflow.\n"
        "- Use clear, human-friendly example utterances under clarify.examples.\n"
        "- Return ONLY the JSON object in a fenced code block."
    )

    return system_message, user_message


def _extract_json_from_response(text: str) -> Dict[str, Any]:
    block = JSON_BLOCK_RE.search(text or "")
    content = block.group(1) if block else text
    return json.loads(content)


@dataclass
class GenerationResult:
    """Container capturing generation metadata."""

    schema: Dict[str, Any]
    raw_response: str
    source: str
    error: Optional[str] = None


def generate_intent_schema(
    system_prompt: str,
    user_prompt: str,
    fallback_context: Dict[str, Any],
) -> GenerationResult:
    """Generate an intent schema using Azure → OpenAI → heuristic."""

    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

    if azure_endpoint and azure_key and azure_deployment:
        try:
            from openai import AzureOpenAI

            client = AzureOpenAI(
                api_key=azure_key,
                api_version=azure_api_version,
                azure_endpoint=azure_endpoint,
            )
            resp = client.chat.completions.create(
                model=azure_deployment,
                temperature=0,
                max_tokens=4096,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw = resp.choices[0].message.content.strip()
            parsed = _extract_json_from_response(raw)
            return GenerationResult(schema=parsed, raw_response=raw, source="azure")
        except Exception as exc:  # pragma: no cover - depends on credentials
            last_error = f"Azure OpenAI error: {exc}"
        else:
            last_error = None
    else:
        last_error = "Azure OpenAI credentials unavailable"

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=openai_key)
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=4096,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw = resp.choices[0].message.content.strip()
            parsed = _extract_json_from_response(raw)
            return GenerationResult(schema=parsed, raw_response=raw, source="openai", error=last_error)
        except Exception as exc:  # pragma: no cover
            last_error = f"OpenAI error: {exc}"
    else:
        last_error = f"{last_error}; OpenAI API key unavailable" if last_error else "OpenAI API key unavailable"

    skeleton = _heuristic_intent_schema(fallback_context)
    return GenerationResult(schema=skeleton, raw_response=json.dumps(skeleton, indent=2), source="heuristic", error=last_error)


def _heuristic_intent_schema(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a deterministic skeleton when no LLM is reachable."""

    workflow = ctx.get("workflow") or {}
    manual_notes = ctx.get("manual_notes", "")
    tools_subset: List[McpTool] = ctx.get("tools_subset", [])

    nodes = workflow.get("nodes", [])
    first_label = (nodes[0].get("label") if nodes else manual_notes.split("\n")[0] if manual_notes else "workflow")
    slug = re.sub(r"[^a-z0-9]+", "-", (first_label or "workflow").lower()).strip("-") or "workflow-intent"

    summary = summarize_workflow(workflow)
    tools_names = [tool.name for tool in tools_subset]

    entities = []
    for tool in tools_subset:
        for arg in tool.args or []:
            if arg.required:
                entities.append(
                    {
                        "name": arg.name,
                        "type": arg.type or "string",
                        "required": True,
                        "extract_rules": "Infer from conversation context.",
                    }
                )
    # Deduplicate entities by name
    dedup: Dict[str, Dict[str, Any]] = {}
    for ent in entities:
        dedup[ent["name"]] = ent
    entities = list(dedup.values())
    if not entities:
        entities = [
            {
                "name": "primary_input",
                "type": "string",
                "required": False,
                "extract_rules": "Capture key detail mentioned by the user.",
            }
        ]

    tasks = []
    for idx, step in enumerate(summary.get("steps") or [first_label], 1):
        node_name = re.sub(r"[^a-z0-9_]+", "_", step.lower()).strip("_") or f"task_{idx}"
        tool_name = tools_names[idx - 1] if idx - 1 < len(tools_names) else None
        task = {
            "task_name": node_name,
            "description": step,
            "type": "action" if tool_name else "process",
        }
        if tool_name:
            task["tool"] = tool_name
            task["inputs"] = [ent["name"] for ent in entities]
        tasks.append(task)

    clarify = {
        "rules": ["Ask politely for missing required details."],
        "ask_when_missing": [ent["name"] for ent in entities if ent.get("required")],
        "ui_hints": ["Keep the assistant responses concise and helpful."],
        "examples": [
            {
                "user": "Example request for the workflow.",
                "entities": {ent["name"]: "" for ent in entities},
            }
        ],
    }

    return {
        "name": slug,
        "summary": manual_notes.strip() or f"Intent for {first_label}",
        "entities": entities,
        "clarify": clarify,
        "task_graph": {
            "nodes": tasks,
            "guidelines": ["Follow the workflow steps sequentially."],
        },
    }


def validate_intent_schema(schema: Dict[str, Any], tools: Sequence[McpTool]) -> List[str]:
    """Validate intent schema shape and semantics."""

    problems: List[str] = []
    if not isinstance(schema, dict):
        return ["Schema must be a JSON object"]

    for key in INTENT_CONTRACT["required"]:
        if key not in schema:
            problems.append(f"Missing required top-level key: {key}")

    entities = schema.get("entities", [])
    if not isinstance(entities, list) or not entities:
        problems.append("`entities` must be a non-empty list")
    else:
        for idx, entity in enumerate(entities):
            if not isinstance(entity, dict):
                problems.append(f"Entity #{idx+1} must be an object")
                continue
            if "name" not in entity or "type" not in entity:
                problems.append(f"Entity '{entity}' missing name/type")

    clarify = schema.get("clarify")
    if not isinstance(clarify, dict):
        problems.append("`clarify` must be an object")
    else:
        for section in ("rules", "ask_when_missing", "ui_hints", "examples"):
            if section not in clarify:
                problems.append(f"Clarify section missing '{section}'")

    task_graph = schema.get("task_graph", {})
    if not isinstance(task_graph, dict):
        problems.append("`task_graph` must be an object")
    else:
        nodes = task_graph.get("nodes")
        if not isinstance(nodes, list) or not nodes:
            problems.append("`task_graph.nodes` must be a non-empty list")
        else:
            tool_lookup = {tool.name: tool for tool in tools}
            entity_names = {ent.get("name") for ent in entities if isinstance(ent, dict)}
            for node in nodes:
                if not isinstance(node, dict):
                    problems.append("Task node must be an object")
                    continue
                for required_key in ("task_name", "type"):
                    if required_key not in node:
                        problems.append(f"Task node missing '{required_key}'")
                tool_name = node.get("tool")
                if tool_name:
                    if tool_name not in tool_lookup:
                        problems.append(f"Task '{node.get('task_name')}' references unknown tool '{tool_name}'")
                        continue
                    required_args = [arg.name for arg in tool_lookup[tool_name].args if arg.required]
                    node_inputs = node.get("inputs", [])
                    if isinstance(node_inputs, dict):
                        node_inputs_iter: Iterable[str] = node_inputs.keys()
                    else:
                        node_inputs_iter = node_inputs
                    input_names = {str(item) for item in node_inputs_iter}
                    for arg in required_args:
                        if arg not in input_names and arg not in entity_names:
                            problems.append(
                                f"Task '{node.get('task_name')}' missing required input '{arg}' for tool '{tool_name}'"
                            )
    return problems


def pick_nearest_sample(
    generated: Dict[str, Any], sample_intents: Sequence[IntentSchema]
) -> Optional[IntentSchema]:
    """Return the sample intent closest to the generated schema."""

    if not sample_intents:
        return None
    target = minify_json(generated)
    best: Tuple[float, Optional[IntentSchema]] = (0.0, None)
    for sample in sample_intents:
        candidate = minify_json(sample.dict())
        ratio = _similarity_ratio(target, candidate)
        if ratio > best[0]:
            best = (ratio, sample)
    return best[1]


def _similarity_ratio(a: str, b: str) -> float:
    import difflib

    return difflib.SequenceMatcher(None, a, b).ratio()


def diff_intent(new_schema: Dict[str, Any], sample_schema: Optional[IntentSchema]) -> Dict[str, Any]:
    """Produce a semantic diff summary between schemas."""

    if sample_schema is None:
        return {"message": "No sample available for comparison."}

    sample_dict = sample_schema.dict()
    summary: Dict[str, Any] = {}

    new_entities = {ent.get("name"): ent for ent in new_schema.get("entities", []) if isinstance(ent, dict)}
    sample_entities = {ent.get("name"): ent for ent in sample_dict.get("entities", []) if isinstance(ent, dict)}

    missing_entities = sorted(set(sample_entities) - set(new_entities))
    extra_entities = sorted(set(new_entities) - set(sample_entities))
    mismatched_types = []
    for name, ent in new_entities.items():
        if name in sample_entities and sample_entities[name].get("type") != ent.get("type"):
            mismatched_types.append(name)

    summary["entity_diff"] = {
        "missing_in_generated": missing_entities,
        "additional_in_generated": extra_entities,
        "type_mismatches": mismatched_types,
    }

    new_tasks = {node.get("task_name"): node for node in new_schema.get("task_graph", {}).get("nodes", []) if isinstance(node, dict)}
    sample_tasks = {node.get("task_name"): node for node in sample_dict.get("task_graph", {}).get("nodes", []) if isinstance(node, dict)}

    missing_tasks = sorted(set(sample_tasks) - set(new_tasks))
    extra_tasks = sorted(set(new_tasks) - set(sample_tasks))
    tool_mismatches = []
    for name, node in new_tasks.items():
        if name in sample_tasks and sample_tasks[name].get("tool") != node.get("tool"):
            tool_mismatches.append(name)

    summary["task_diff"] = {
        "missing_in_generated": missing_tasks,
        "additional_in_generated": extra_tasks,
        "tool_mismatches": tool_mismatches,
    }

    summary["text_diff"] = "\n".join(
        difflib.unified_diff(
            json.dumps(sample_dict, indent=2).splitlines(),
            json.dumps(new_schema, indent=2).splitlines(),
            fromfile="sample",
            tofile="generated",
            lineterm="",
        )
    )
    return summary


def dry_run_intent(schema: Dict[str, Any], tools: Sequence[McpTool]) -> Dict[str, Any]:
    """Dry-run analysis mapping tasks to tools and validating args."""

    if not schema:
        return {"status": "no-schema"}

    tool_lookup = {tool.name: tool for tool in tools}
    entity_names = {ent.get("name") for ent in schema.get("entities", []) if isinstance(ent, dict)}
    issues: List[str] = []
    task_checks: List[Dict[str, Any]] = []

    for node in schema.get("task_graph", {}).get("nodes", []) or []:
        tool_name = node.get("tool")
        required_args: List[str] = []
        if tool_name and tool_name in tool_lookup:
            required_args = [arg.name for arg in tool_lookup[tool_name].args if arg.required]
            provided_inputs: Iterable[str]
            inputs = node.get("inputs", [])
            if isinstance(inputs, dict):
                provided_inputs = inputs.keys()
            else:
                provided_inputs = inputs
            provided = set(str(item) for item in provided_inputs)
            missing = [arg for arg in required_args if arg not in provided and arg not in entity_names]
            if missing:
                issues.append(
                    f"Task '{node.get('task_name')}' missing required inputs: {', '.join(missing)}"
                )
        else:
            if tool_name and tool_name not in tool_lookup:
                issues.append(f"Task '{node.get('task_name')}' references unknown tool '{tool_name}'")
        task_checks.append({
            "task": node.get("task_name"),
            "tool": tool_name,
            "required_args": required_args,
        })

    status = "ok" if not issues else "issues"
    return {"status": status, "issues": issues, "tasks": task_checks}


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure the given directory exists and return it as Path."""

    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def git_commit(files: Sequence[Union[str, Path]], message: str) -> Tuple[bool, str]:
    """Attempt to create a git commit with the provided files."""

    paths = [str(Path(f)) for f in files]
    try:
        subprocess.run(["git", "add", *paths], check=True, capture_output=True)
        result = subprocess.run(
            ["git", "commit", "-m", message], check=True, capture_output=True, text=True
        )
        return True, result.stdout.strip()
    except subprocess.CalledProcessError as exc:
        return False, exc.stderr.strip() or str(exc)


__all__ = [
    "McpArg",
    "McpTool",
    "IntentEntity",
    "IntentTaskNode",
    "IntentSchema",
    "INTENT_CONTRACT",
    "GenerationResult",
    "load_tools",
    "load_intents",
    "minify_json",
    "choose_candidate_tools",
    "summarize_workflow",
    "compose_prompt",
    "generate_intent_schema",
    "validate_intent_schema",
    "pick_nearest_sample",
    "diff_intent",
    "dry_run_intent",
    "ensure_directory",
    "git_commit",
]
