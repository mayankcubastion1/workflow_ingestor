import json
import sys
import types
from pathlib import Path


if "networkx" not in sys.modules:
    class _FakeDiGraph:
        def __init__(self):
            self._nodes = {}
            self._edges = {}

        def add_node(self, node_id, **data):
            self._nodes[node_id] = data

        def add_edge(self, src, tgt, **data):
            self._edges.setdefault(src, []).append((tgt, data))

        def successors(self, node_id):
            return [target for target, _ in self._edges.get(node_id, [])]

        def get_edge_data(self, src, tgt):
            for target, payload in self._edges.get(src, []):
                if target == tgt:
                    return payload
            return {}

        @property
        def nodes(self):
            return self._nodes.keys()

    fake_networkx = types.SimpleNamespace(
        DiGraph=_FakeDiGraph,
        find_cycle=lambda *args, **kwargs: [],
        exception=types.SimpleNamespace(NetworkXNoCycle=Exception),
    )
    sys.modules["networkx"] = fake_networkx


if "pydantic" not in sys.modules:
    class _BaseModel:
        def __init__(self, **data):
            for key, value in data.items():
                setattr(self, key, value)

        def dict(self):
            return dict(self.__dict__)

    def _field(default=None, default_factory=None, **_kwargs):
        if default_factory is not None:
            return default_factory()
        return default

    fake_pydantic = types.SimpleNamespace(BaseModel=_BaseModel, Field=_field)
    sys.modules["pydantic"] = fake_pydantic


sys.path.append(str(Path(__file__).resolve().parents[1]))

from intent_generator_utils import (
    INTENT_CONTRACT,
    IntentSchema,
    McpArg,
    McpTool,
    choose_candidate_tools,
    compose_prompt,
    diff_intent,
    dry_run_intent,
    minify_json,
    validate_intent_schema,
)


def test_minify_json_removes_whitespace():
    obj = {"name": "Example", "values": [1, 2, 3]}
    compact = minify_json(obj)
    assert compact == json.dumps(obj, separators=(",", ":"))


def test_choose_candidate_tools_prefers_named_tool():
    tools = [
        McpTool(
            name="query_account",
            description="Look up account information",
            args=[McpArg(name="account_id", type="string", required=True)],
            execution={"endpoint": ""},
        ),
        McpTool(
            name="update_status",
            description="Update ticket status",
            args=[],
            execution={"endpoint": ""},
        ),
    ]
    workflow = {
        "direction": "TD",
        "nodes": [
            {"id": "A", "label": "Query Account", "type": "process"},
            {"id": "B", "label": "Update Status", "type": "process"},
        ],
        "edges": [],
    }
    subset = choose_candidate_tools(workflow, "", tools, top_k=1)
    assert subset and subset[0].name == "query_account"


def test_validate_intent_schema_detects_unknown_tool():
    tools = [
        McpTool(
            name="known_tool",
            description="Performs known action",
            args=[],
            execution={},
        )
    ]
    schema = {
        "name": "sample_intent",
        "summary": "Test",
        "entities": [{"name": "account_id", "type": "string"}],
        "clarify": {"rules": [], "ask_when_missing": [], "ui_hints": [], "examples": []},
        "task_graph": {
            "nodes": [
                {
                    "task_name": "unknown_task",
                    "description": "Uses missing tool",
                    "type": "action",
                    "tool": "missing_tool",
                }
            ]
        },
    }
    problems = validate_intent_schema(schema, tools)
    assert any("missing_tool" in p for p in problems)


def test_dry_run_intent_reports_missing_inputs():
    tools = [
        McpTool(
            name="fetch_data",
            description="Fetch data",
            args=[McpArg(name="record_id", type="string", required=True)],
            execution={},
        )
    ]
    schema = {
        "name": "fetch",
        "summary": "Fetch record",
        "entities": [],
        "clarify": {"rules": [], "ask_when_missing": [], "ui_hints": [], "examples": []},
        "task_graph": {
            "nodes": [
                {
                    "task_name": "pull",
                    "description": "Fetches the record",
                    "type": "action",
                    "tool": "fetch_data",
                    "inputs": [],
                }
            ]
        },
    }
    dry_run = dry_run_intent(schema, tools)
    assert dry_run["status"] == "issues"
    assert any("record_id" in issue for issue in dry_run["issues"])


def test_compose_prompt_contains_contract_section():
    tools = [
        McpTool(
            name="do_work",
            description="Perform work",
            args=[],
            execution={},
        )
    ]
    workflow = {
        "direction": "TD",
        "nodes": [{"id": "S", "label": "Start", "type": "start"}],
        "edges": [],
    }
    system, user = compose_prompt(workflow, "", tools, [], INTENT_CONTRACT)
    assert "STRICT OUTPUT CONTRACT" in system
    assert "WORKFLOW_CONTEXT" in user


def test_diff_intent_reports_entity_changes():
    base_schema = {
        "name": "intent",
        "summary": "summary",
        "entities": [{"name": "a", "type": "string"}],
        "clarify": {"rules": [], "ask_when_missing": [], "ui_hints": [], "examples": []},
        "task_graph": {"nodes": []},
    }
    generated = {
        "name": "intent",
        "summary": "summary",
        "entities": [{"name": "b", "type": "string"}],
        "clarify": {"rules": [], "ask_when_missing": [], "ui_hints": [], "examples": []},
        "task_graph": {"nodes": []},
    }
    diff = diff_intent(generated, None)
    assert "message" in diff
    diff = diff_intent(generated, IntentSchema(**base_schema))
    assert diff["entity_diff"]["missing_in_generated"] == ["a"]
