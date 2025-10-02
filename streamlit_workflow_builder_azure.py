"""
Workflow Builder: Natural Language ↔ Diagram (Streamlit)

Quick start:
  pip install -U streamlit pydantic pyyaml networkx graphviz streamlit-mermaid openai python-dotenv
  streamlit run streamlit_workflow_builder_azure_fixed.py

Notes:
- LLM is optional. If Azure OpenAI (AZURE_OPENAI_*) or OPENAI_API_KEY is set, the app will ask an LLM to convert text → Mermaid.
- Without an LLM key, the app falls back to a simple heuristic that makes a linear flow from bullet points.
- Diagrams are rendered with Mermaid and can be exported as Mermaid/JSON/YAML.
- Users can also build/edit DAGs via a point‑and‑click Node/Edge editor.
"""
from __future__ import annotations

import os
import re
import io
import json
import yaml
import textwrap
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Load .env for credentials (Azure/OpenAI)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import streamlit as st
import streamlit.components.v1 as components
from pydantic import BaseModel, Field, validator
import networkx as nx
import pandas as pd

from intent_generator_utils import (
    GenerationResult,
    INTENT_CONTRACT,
    IntentSchema,
    McpTool,
    choose_candidate_tools,
    compose_prompt,
    diff_intent,
    dry_run_intent,
    ensure_directory,
    generate_intent_schema,
    git_commit,
    load_intents,
    load_tools,
    pick_nearest_sample,
    validate_intent_schema,
)

# Attempt to import the Mermaid component
try:
    from streamlit_mermaid import st_mermaid
    _HAS_MERMAID = True
except Exception:
    _HAS_MERMAID = False

# ---------------------------
# Data model
# ---------------------------

NODE_TYPES = [
    "start", "process", "decision", "io", "subroutine", "end"
]

@dataclass
class Node:
    id: str
    label: str
    type: str = "process"  # one of NODE_TYPES

@dataclass
class Edge:
    source: str
    target: str
    label: str = ""

class Workflow(BaseModel):
    nodes: List[Dict] = Field(default_factory=list)
    edges: List[Dict] = Field(default_factory=list)
    direction: str = Field(default="TD", description="Mermaid direction: TD, LR, BT, RL")

    @validator("direction")
    def _valid_dir(cls, v: str) -> str:
        if v not in {"TD", "LR", "BT", "RL"}:
            raise ValueError("direction must be TD, LR, BT, or RL")
        return v

    def as_nodes(self) -> List[Node]:
        return [Node(**n) for n in self.nodes]

    def as_edges(self) -> List[Edge]:
        return [Edge(**e) for e in self.edges]

# ---------------------------
# Helpers for editors / validation / fallback rendering
# (Placed above UI usage to avoid NameError on streamlit reruns)
# ---------------------------

def _nodes_to_dataframe(nodes: List[Node]) -> pd.DataFrame:
    return pd.DataFrame([{"id": n.id, "label": n.label, "type": n.type} for n in nodes])


def _edges_to_dataframe(edges: List[Edge]) -> pd.DataFrame:
    return pd.DataFrame([{"source": e.source, "target": e.target, "label": e.label} for e in edges])


def _dataframe_to_nodes(df: pd.DataFrame) -> List[Node]:
    out: List[Node] = []
    for _, row in df.iterrows():
        nid = str(row.get("id", "").strip())
        if not nid:
            raise ValueError("Node id cannot be empty")
        nlabel = str(row.get("label", nid))
        ntype = str(row.get("type", "process"))
        if ntype not in NODE_TYPES:
            raise ValueError(f"Invalid node type: {ntype}")
        out.append(Node(id=nid, label=nlabel, type=ntype))
    # ensure unique ids
    ids = [n.id for n in out]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate node IDs detected")
    return out


def _dataframe_to_edges(df: pd.DataFrame) -> List[Edge]:
    out: List[Edge] = []
    for _, row in df.iterrows():
        src = str(row.get("source", "").strip())
        tgt = str(row.get("target", "").strip())
        if not src or not tgt:
            raise ValueError("Edge source/target cannot be empty")
        lab = str(row.get("label", ""))
        out.append(Edge(source=src, target=tgt, label=lab))
    return out


def _mk_graph(nodes: List[Node], edges: List[Edge]) -> nx.DiGraph:
    g = nx.DiGraph()
    for n in nodes:
        g.add_node(n.id, label=n.label, type=n.type)
    for e in edges:
        g.add_edge(e.source, e.target, label=e.label)
    return g


def _validate_workflow(nodes: List[Node], edges: List[Edge]) -> Tuple[bool, List[str]]:
    problems: List[str] = []
    ids = {n.id for n in nodes}
    for e in edges:
        if e.source not in ids:
            problems.append(f"Edge source '{e.source}' missing from nodes")
        if e.target not in ids:
            problems.append(f"Edge target '{e.target}' missing from nodes")
    g = _mk_graph(nodes, edges)
    try:
        cycle = nx.find_cycle(g, orientation="original")
        if cycle:
            problems.append("Graph contains a cycle (DAG required).")
    except nx.exception.NetworkXNoCycle:
        pass
    # Check single start/end presence (optional, but nice UX)
    starts = [n for n in nodes if n.type == "start"]
    ends = [n for n in nodes if n.type == "end"]
    if len(starts) > 1:
        problems.append("Multiple 'start' nodes found.")
    if len(ends) > 1:
        problems.append("Multiple 'end' nodes found.")
    return (len(problems) == 0, problems)


def _mermaid_as_graphviz_dot(mermaid: str) -> str:
    wf = mermaid_to_workflow(mermaid)
    nodes = wf.as_nodes()
    edges = wf.as_edges()
    buf = ["digraph G {", "rankdir=" + ("LR" if wf.direction in ("LR", "RL") else "TB") + ";"]
    # Node shapes mapping (use Graphviz-valid shapes)
    shape_map = {
        "process": "box",
        "decision": "diamond",
        "io": "ellipse",
        "subroutine": "box3d",
        "start": "ellipse",
        "end": "ellipse",
    }
    for n in nodes:
        label = n.label.replace("\n", "\\n")
        shape = shape_map.get(n.type, "box")
        buf.append(f'  {n.id} [label="{label}", shape={shape}]')
    for e in edges:
        lab = f' [label="{e.label}"]' if e.label else ""
        buf.append(f"  {e.source} -> {e.target}{lab}")
    buf.append("}")
    return "\n".join(buf)

# ---------------------------
# Conversions: Workflow ↔ Mermaid
# ---------------------------

# (kept for potential future use)
SHAPE_OPEN = {
    "process": "[",
    "decision": "{",
    "io": "(",
    "subroutine": "[[",
    "start": "([",
    "end": "])",  # end uses close partner below
}

SHAPE_CLOSE = {
    "process": "]",
    "decision": "}",
    "io": ")",
    "subroutine": "]]",
    "start": ")]",
    "end": "([",  # swapped on purpose when we render end as rounded
}

# For end we prefer (()) rounded; we handle it specially in renderer.

def workflow_to_mermaid(wf: Workflow) -> str:
    nodes = wf.as_nodes()
    edges = wf.as_edges()
    lines = [f"flowchart {wf.direction}"]
    for n in nodes:
        label = n.label.replace("\n", "\\n")
        # Choose shape
        if n.type == "decision":
            shape_open, shape_close = "{", "}"
        elif n.type in ("start", "end"):
            shape_open, shape_close = "(", ")"
        elif n.type == "subroutine":
            shape_open, shape_close = "[[", "]]"
        elif n.type == "io":
            shape_open, shape_close = "(", ")"
        else:
            shape_open, shape_close = "[", "]"
        lines.append(f"    {n.id}{shape_open}{label}{shape_close}")
    for e in edges:
        label = f"|{e.label}|" if e.label else ""
        lines.append(f"    {e.source} --{label}--> {e.target}")
    return "\n".join(lines)

NODE_DECL_RE = re.compile(r"^\s*([A-Za-z0-9_\-]+)\s*(\[|\(|\{|\[\[)")
EDGE_RE = re.compile(r"^\s*([A-Za-z0-9_\-]+)\s*--(?:\|([^|]+)\|)?-->\s*([A-Za-z0-9_\-]+)")
LABEL_EXTRACT_RE = re.compile(r"^\s*[A-Za-z0-9_\-]+\s*(\[\[|\[|\(|\{|\(\[)\s*(.*?)\s*(\]\]|\]|\)|\}|\]\))\s*$")

SHAPE_TO_TYPE = {
    "[": "process",
    "(": "process",  # will adjust below based on lookback
    "{": "decision",
    "[[": "subroutine",
}

CLOSE_TO_TYPE_HINT = {
    "]": "process",
    ")": "io",  # treat () as IO unless start/end found elsewhere
    "}": "decision",
    "]]": "subroutine",
    ")]": "end",
}

def mermaid_to_workflow(mermaid: str, direction_fallback: str = "TD") -> Workflow:
    lines = [ln.strip() for ln in mermaid.splitlines() if ln.strip()]
    dir_ = direction_fallback
    if lines and lines[0].lower().startswith("flowchart "):
        try:
            dir_ = lines[0].split()[1].strip().upper()
            lines = lines[1:]
        except Exception:
            pass

    nodes: Dict[str, Node] = {}
    edges: List[Edge] = []

    # First pass: detect node declarations
    for ln in lines:
        if "--" in ln:
            continue
        m = LABEL_EXTRACT_RE.match(ln)
        if m:
            open_tok, label, close_tok = m.groups()
            node_id_match = re.match(r"([A-Za-z0-9_\-]+)", ln)
            if not node_id_match:
                continue
            node_id = node_id_match.group(1)
            ntype = CLOSE_TO_TYPE_HINT.get(close_tok, SHAPE_TO_TYPE.get(open_tok, "process"))
            # Heuristic: if label looks like "Start" or "End" set type accordingly
            low = label.lower()
            if "start" == low or low.startswith("start "):
                ntype = "start"
            if low == "end" or low.startswith("end "):
                ntype = "end"
            nodes[node_id] = Node(id=node_id, label=label, type=ntype)

    # Second pass: edges
    for ln in lines:
        m = EDGE_RE.match(ln)
        if not m:
            continue
        src, lab, tgt = m.groups()
        edges.append(Edge(source=src, target=tgt, label=(lab or "").strip()))
        # Create implicit nodes if not declared
        if src not in nodes:
            nodes[src] = Node(id=src, label=src)
        if tgt not in nodes:
            nodes[tgt] = Node(id=tgt, label=tgt)

    wf = Workflow(
        nodes=[asdict(n) for n in nodes.values()],
        edges=[asdict(e) for e in edges],
        direction=dir_,
    )
    return wf

# ---------------------------
# NL → Mermaid via LLM (optional)
# ---------------------------

DEFAULT_SYSTEM_PROMPT = textwrap.dedent(
    """
    You convert natural language workflow descriptions into a **Mermaid flowchart**. Output ONLY a valid Mermaid code block.

    Rules:
    - Start with: `flowchart TD` (or `LR` if order benefits left→right).
    - Use concise alphanumeric IDs (e.g., S, A1, D1) and readable labels.
    - Use decision diamonds for conditionals: `{question?}` with yes/no branches.
    - Use rounded nodes for START/END.
    - Use labels on edges where helpful: `A -- yes --> B`.
    - Avoid extraneous prose—respond ONLY with Mermaid code fenced in triple backticks.

    Example:
    ```
    flowchart TD
      S((Start)) --> A[Load data]
      A --> D{Valid?}
      D -- yes --> B[Transform]
      D -- no --> E[Log error]
      B --> Z((End))
      E --> Z
    ```
    """
)

# Fixed: made template parameterizable (no stray "last" placeholder left over)
MERMAID_FALLBACK_TEMPLATE = textwrap.dedent(
    """
    flowchart TD
    S((Start))
    Z((End))
    {body}
    S --> step1
    {edges}
    {last} --> Z
    """
)

BULLET_RE = re.compile(r"^\s*[-*\d+.)]+\s*(.+)$")


def _heuristic_nl_to_mermaid(text: str) -> str:
    # Turn bullet/numbered lines into a simple linear flow.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    steps = []
    for ln in lines:
        m = BULLET_RE.match(ln)
        if m:
            steps.append(m.group(1))
        elif len(ln.split()) > 2:  # treat open text lines as steps too
            steps.append(ln)
    if not steps:
        steps = ["Process input", "Do work", "Produce output"]
    node_lines = []
    edge_lines = []
    last_id = "S"
    for i, step in enumerate(steps, 1):
        nid = f"step{i}"
        node_lines.append(f"{nid}[{step}]")
        edge_lines.append(f"{last_id} --> {nid}")
        last_id = nid
    body = "\n".join(f"  {x}" for x in node_lines)
    edges = "\n".join(f"  {x}" for x in edge_lines)
    return MERMAID_FALLBACK_TEMPLATE.format(body=body, edges=edges, last=last_id)

# Prefer Azure OpenAI, then fall back to public OpenAI, else heuristic

def nl_to_mermaid_via_llm(text: str, system_prompt: str) -> Tuple[str, Optional[str]]:
    """Returns (mermaid, error). Priority: Azure OpenAI → OpenAI → heuristic."""
    # --- Azure OpenAI first ---
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")  # deployment name, e.g. "gpt-4o"
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
                model=azure_deployment,  # deployment name
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
            )
            raw = resp.choices[0].message.content.strip()
            code = extract_mermaid_block(raw)
            if not code:
                raise ValueError("LLM did not return a Mermaid code block.")
            return code, None
        except Exception as e:
            return _heuristic_nl_to_mermaid(text), f"Azure OpenAI error: {e}. Falling back to heuristic."

    # --- Public OpenAI fallback ---
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return _heuristic_nl_to_mermaid(text), "No LLM credentials found; using heuristic conversion."
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
        )
        raw = resp.choices[0].message.content.strip()
        code = extract_mermaid_block(raw)
        if not code:
            raise ValueError("LLM did not return a Mermaid code block.")
        return code, None
    except Exception as e:
        return _heuristic_nl_to_mermaid(text), f"LLM error: {e}. Falling back to heuristic."

MERMAID_BLOCK_RE = re.compile(r"```(?:mermaid)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


def extract_mermaid_block(text: str) -> Optional[str]:
    m = MERMAID_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()
    # Also accept raw mermaid (no fences)
    if text.startswith("flowchart "):
        return text.strip()
    return None

# ---------------------------
# Streamlit App
# ---------------------------

st.set_page_config(page_title="Workflow Builder: NL ↔ Diagram", layout="wide")

if "wf" not in st.session_state:
    # Minimal starter workflow
    starter = Workflow(
        nodes=[
            {"id": "S", "label": "Start", "type": "start"},
            {"id": "A", "label": "Do something", "type": "process"},
            {"id": "Z", "label": "End", "type": "end"},
        ],
        edges=[{"source": "S", "target": "A", "label": ""}, {"source": "A", "target": "Z", "label": ""}],
        direction="TD",
    )
    st.session_state.wf = starter

st.title("⚙️ Workflow Builder: Natural Language ↔ Diagram")

with st.sidebar:
    st.subheader("Settings")
    st.caption("LLM is optional. Set Azure OpenAI (AZURE_OPENAI_*) or OPENAI_API_KEY to enable NL→diagram.")
    direction = st.selectbox("Diagram direction", ["TD", "LR", "BT", "RL"], index=["TD","LR","BT","RL"].index(st.session_state.wf.direction))
    st.session_state.wf.direction = direction
    with st.expander("System prompt (for NL→diagram)", expanded=False):
        sys_prompt = st.text_area("", value=DEFAULT_SYSTEM_PROMPT, height=260)
    st.markdown("---")
    st.subheader("Import / Export")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.download_button("⬇️ JSON", data=json.dumps(st.session_state.wf.dict(), indent=2).encode(), file_name="workflow.json", mime="application/json", key="dl_json")
    with col_b:
        st.download_button("⬇️ YAML", data=yaml.safe_dump(st.session_state.wf.dict()).encode(), file_name="workflow.yaml", mime="text/yaml", key="dl_yaml")
    with col_c:
        mer = workflow_to_mermaid(st.session_state.wf)
        st.download_button("⬇️ Mermaid", data=mer.encode(), file_name="workflow.mmd", mime="text/plain", key="dl_mmd")
    up = st.file_uploader("Import JSON or YAML", type=["json", "yaml", "yml"], key="uploader")
    if up is not None:
        try:
            payload = up.read().decode("utf-8")
            data = json.loads(payload) if up.name.endswith(".json") else yaml.safe_load(payload)
            st.session_state.wf = Workflow(**data)
            st.success("Imported workflow.")
        except Exception as e:
            st.error(f"Import failed: {e}")

mode = st.tabs([
    "📝 Natural language",
    "🧩 Node/Edge editor",
    "🧭 Mermaid editor",
    "🧠 Intent Schema Generator",
    "▶️ Simulate",
])

# ---------------------------
# Tab 1: Natural language → Mermaid
# ---------------------------
with mode[0]:
    st.subheader("Describe your workflow")
    nl_text = st.text_area("Describe the pipeline (bullets or prose)", height=220, placeholder="e.g. Ingest CSV → Validate schema → If invalid, log and stop; otherwise clean data → Load to warehouse → Notify Slack...", key="nl_text")
    generate = st.button("✨ Generate diagram", use_container_width=True, key="gen_btn")
    if generate and nl_text.strip():
        mermaid, err = nl_to_mermaid_via_llm(nl_text.strip(), sys_prompt)
        if err:
            st.warning(err)
        st.session_state.last_mermaid = mermaid
        wf = mermaid_to_workflow(mermaid, direction_fallback=st.session_state.wf.direction)
        st.session_state.wf = wf

    if "last_mermaid" in st.session_state:
        st.markdown("**Generated Mermaid:**")
        st.code(st.session_state.last_mermaid, language="mermaid")

    st.markdown("**Preview:**")
    mer = workflow_to_mermaid(st.session_state.wf)
    if _HAS_MERMAID:
        # FIX: provide unique key to avoid StreamlitDuplicateElementId
        st_mermaid(mer, height=480, key="mermaid_preview_tab1")
    else:
        st.graphviz_chart(_mermaid_as_graphviz_dot(mer))
        st.info("Install `streamlit-mermaid` for better rendering.")

# ---------------------------
# Tab 2: Node/Edge editor
# ---------------------------
with mode[1]:
    st.subheader("Edit nodes and edges")
    ndf = _nodes_to_dataframe(st.session_state.wf.as_nodes())
    edf = _edges_to_dataframe(st.session_state.wf.as_edges())
    st.caption("Tip: IDs must be unique. Types: start, process, decision, io, subroutine, end.")
    ndf = st.data_editor(ndf, num_rows="dynamic", use_container_width=True, key="nodes_editor")
    edf = st.data_editor(edf, num_rows="dynamic", use_container_width=True, key="edges_editor")

    if st.button("Update diagram from tables", type="primary", key="update_from_tables"):
        try:
            nodes = _dataframe_to_nodes(ndf)
            edges = _dataframe_to_edges(edf)
            ok, issues = _validate_workflow(nodes, edges)
            st.session_state.wf = Workflow(
                nodes=[asdict(n) for n in nodes],
                edges=[asdict(e) for e in edges],
                direction=st.session_state.wf.direction,
            )
            if ok:
                st.success("Updated.")
            else:
                for p in issues:
                    st.warning(p)
        except Exception as e:
            st.error(f"Validation failed: {e}")

    st.markdown("**Preview:**")
    mer = workflow_to_mermaid(st.session_state.wf)
    if _HAS_MERMAID:
        # FIX: unique key
        st_mermaid(mer, height=520, key="mermaid_preview_tab2")
    else:
        st.graphviz_chart(_mermaid_as_graphviz_dot(mer))

# ---------------------------
# Tab 3: Mermaid editor
# ---------------------------
with mode[2]:
    st.subheader("Directly edit Mermaid")
    curr = workflow_to_mermaid(st.session_state.wf)
    mermaid_text = st.text_area("Mermaid flowchart", value=curr, height=320, key="mermaid_editor")
    convert = st.button("Apply from Mermaid", key="apply_mermaid")
    if convert:
        try:
            wf = mermaid_to_workflow(mermaid_text, direction_fallback=st.session_state.wf.direction)
            st.session_state.wf = wf
            st.success("Applied Mermaid → workflow.")
        except Exception as e:
            st.error(f"Parse failed: {e}")
    st.markdown("**Preview:**")
    if _HAS_MERMAID:
        # FIX: unique key (this was the stacktrace site)
        st_mermaid(mermaid_text, height=520, key="mermaid_preview_tab3")
    else:
        st.graphviz_chart(_mermaid_as_graphviz_dot(mermaid_text))

# ---------------------------
# Tab 4: Intent Schema Generator
# ---------------------------
with mode[3]:
    st.subheader("🧠 Intent Schema Generator")
    workflow_dict = st.session_state.wf.dict()
    sample_objects: List[IntentSchema] = []
    sample_labels: List[str] = []
    tools: List[McpTool] = []
    candidate_tools: List[McpTool] = []
    system_prompt = ""
    user_prompt = ""
    manual_notes = ""

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("### 1. Source inputs")
        samples_dir = Path("./sample jsons")
        available_samples = sorted(str(p) for p in samples_dir.glob("*.json"))
        default_sample_paths = st.session_state.get("intent_sample_paths", available_samples[:2])
        default_sample_paths = [p for p in default_sample_paths if p in available_samples]
        selected_sample_paths = st.multiselect(
            "Sample intent JSONs",
            options=available_samples,
            default=default_sample_paths,
            help="Select existing gold intents to ground the LLM output.",
        )

        sample_objects = []
        sample_labels = []
        if selected_sample_paths:
            try:
                loaded_samples = load_intents(selected_sample_paths)
                sample_objects.extend(loaded_samples)
                sample_labels.extend(selected_sample_paths)
                st.session_state.intent_sample_paths = selected_sample_paths
            except Exception as exc:
                st.error(str(exc))

        uploaded_samples = st.file_uploader(
            "Upload additional sample intents",
            type=["json"],
            accept_multiple_files=True,
            key="intent_sample_upload",
        )
        if uploaded_samples:
            for uploaded in uploaded_samples:
                try:
                    payload = json.loads(uploaded.getvalue().decode("utf-8"))
                    sample = IntentSchema(**payload)
                    sample_objects.append(sample)
                    sample_labels.append(f"uploaded:{uploaded.name}")
                except Exception as exc:
                    st.error(f"Failed to parse uploaded sample '{uploaded.name}': {exc}")

        st.markdown("---")
        st.markdown("### Tools catalog")
        default_tools_path = st.session_state.get("intent_tools_path")
        if not default_tools_path:
            guess_tools_path = samples_dir / "tools.json"
            default_tools_path = str(guess_tools_path) if guess_tools_path.exists() else ""
        tools_path_input = st.text_input(
            "Path to tools.json",
            value=default_tools_path,
            help="Load the MCP tool catalog used by the agent runtime.",
            key="intent_tools_path_input",
        )
        tools_upload = st.file_uploader(
            "Or upload tools.json",
            type=["json"],
            accept_multiple_files=False,
            key="intent_tools_upload",
        )

        tools = []
        tools_source = None
        if tools_upload is not None:
            try:
                payload = json.loads(tools_upload.getvalue().decode("utf-8"))
                if isinstance(payload, dict) and "tools" in payload:
                    payload = payload["tools"]
                if not isinstance(payload, list):
                    raise ValueError("Expected a list of tool definitions.")
                tools = [McpTool(**item) for item in payload]
                tools_source = f"uploaded:{tools_upload.name}"
                st.caption(f"Loaded {len(tools)} tools from upload {tools_upload.name}.")
            except Exception as exc:
                tools = []
                st.error(f"Failed to load tools from upload: {exc}")
        elif tools_path_input:
            try:
                tools = load_tools(tools_path_input)
                tools_source = tools_path_input
                st.session_state.intent_tools_path = tools_path_input
                st.caption(f"Loaded {len(tools)} tools from `{tools_path_input}`.")
            except Exception as exc:
                tools = []
                st.error(f"Failed to load tools: {exc}")
        else:
            st.info("Provide a tools.json catalog to enable schema validation.")

        st.markdown("---")
        st.markdown("### Workflow source")
        use_current_graph = st.toggle(
            "Use current Workflow DAG as input",
            value=st.session_state.get("intent_use_graph", True),
            key="intent_use_graph_toggle",
        )
        st.session_state.intent_use_graph = use_current_graph

        manual_description = ""
        if not use_current_graph:
            manual_description = st.text_area(
                "Describe workflow",
                value=st.session_state.get("intent_manual_description", ""),
                height=200,
            )
            st.session_state.intent_manual_description = manual_description
        additional_notes = st.text_area(
            "Additional notes for the LLM (optional)",
            value=st.session_state.get("intent_additional_notes", ""),
            height=140,
        )
        st.session_state.intent_additional_notes = additional_notes
        manual_notes = "\n".join(part for part in [manual_description, additional_notes] if part).strip()

        workflow_for_prompt = workflow_dict if use_current_graph else None
        if tools:
            candidate_tools = choose_candidate_tools(workflow_for_prompt, manual_notes, tools)
        else:
            candidate_tools = []

        system_prompt, user_prompt = compose_prompt(
            workflow_for_prompt,
            manual_notes,
            candidate_tools,
            sample_objects,
            INTENT_CONTRACT,
        )

        st.markdown("### 2. Prompt composer (preview)")
        st.caption("Exact payload sent to the LLM. Read-only preview.")
        prompt_preview = f"System message:\n{system_prompt}\n\nUser message:\n{user_prompt}"
        st.code(prompt_preview, language="markdown")

        st.markdown("### 3. Generate Intent Schema")
        if not tools:
            st.warning("Load a tools.json catalog to enable generation and validation.")
        fallback_context = {
            "workflow": workflow_for_prompt,
            "manual_notes": manual_notes,
            "tools_subset": candidate_tools,
        }
        if st.button(
            "Generate Intent Schema",
            type="primary",
            use_container_width=True,
            disabled=not tools,
            key="intent_generate_btn",
        ):
            try:
                result = generate_intent_schema(system_prompt, user_prompt, fallback_context)
                st.session_state.generated_intent_schema = result.schema
                st.session_state.generated_intent_raw = result.raw_response
                st.session_state.generated_intent_source = result.source
                st.session_state.generated_intent_error = result.error
                st.session_state.generated_intent_validation = (
                    validate_intent_schema(result.schema, tools) if tools else []
                )
                st.session_state.generated_intent_samples = [label for label in sample_labels]
                st.session_state.generated_intent_manual_notes = manual_notes
                st.session_state.generated_intent_last_tools = [tool.dict() for tool in tools]
                st.session_state.last_intent_save_path = None
                st.success(f"Generated schema via {result.source} backend.")
                if result.error:
                    st.info(result.error)
            except Exception as exc:
                st.error(f"Generation failed: {exc}")

    with right:
        st.markdown("### Validated JSON")
        generated_schema = st.session_state.get("generated_intent_schema")
        if generated_schema:
            pretty_json = json.dumps(generated_schema, indent=2, ensure_ascii=False)
            validation_messages = (
                validate_intent_schema(generated_schema, tools) if tools else st.session_state.get("generated_intent_validation", [])
            )
            if validation_messages:
                st.markdown("**Validation:** ❌ Issues found")
                for msg in validation_messages:
                    st.warning(msg)
            else:
                st.markdown("**Validation:** ✅ Passed")
            st.json(generated_schema)
            source = st.session_state.get("generated_intent_source", "unknown")
            st.caption(f"Generated via: {source}")
            if st.session_state.get("generated_intent_error"):
                st.info(st.session_state["generated_intent_error"])

            nearest_sample = (
                pick_nearest_sample(generated_schema, sample_objects) if sample_objects else None
            )
            if nearest_sample:
                label_map = {id(obj): label for label, obj in zip(sample_labels, sample_objects)}
                nearest_label = label_map.get(id(nearest_sample), "sample")
                diff_summary = diff_intent(generated_schema, nearest_sample)
                st.markdown(f"### Diff vs. `{nearest_label}`")
                col_gen, col_sample = st.columns(2)
                with col_gen:
                    st.caption("Generated intent")
                    st.code(pretty_json, language="json")
                with col_sample:
                    st.caption(f"Sample intent ({nearest_label})")
                    st.code(
                        json.dumps(nearest_sample.dict(), indent=2, ensure_ascii=False),
                        language="json",
                    )
                st.markdown("**Entity diff**")
                st.json(diff_summary.get("entity_diff"))
                st.markdown("**Task diff**")
                st.json(diff_summary.get("task_diff"))
                if diff_summary.get("text_diff"):
                    st.markdown("**Unified diff**")
                    st.code(diff_summary["text_diff"], language="diff")
            else:
                st.info("Select one or more sample intents to view a diff.")

            st.markdown("### Export & Actions")
            export_col1, export_col2, export_col3 = st.columns(3)
            export_col1.download_button(
                "⬇️ Export JSON",
                data=pretty_json.encode("utf-8"),
                file_name=f"{generated_schema.get('name', 'intent')}.json",
                mime="application/json",
                key="intent_download_btn",
            )
            if export_col2.button("📋 Copy to Clipboard", key="intent_copy_btn"):
                components.html(
                    f"""
                    <script>
                    const text = {json.dumps(pretty_json)};
                    navigator.clipboard.writeText(text);
                    </script>
                    """,
                    height=0,
                )
                st.toast("Copied intent JSON to clipboard")
            if export_col3.button("💾 Save to ./intents", key="intent_save_btn"):
                try:
                    target_dir = ensure_directory("./intents")
                    target_path = target_dir / f"{generated_schema.get('name', 'intent')}.json"
                    target_path.write_text(pretty_json + "\n", encoding="utf-8")
                    st.session_state.last_intent_save_path = str(target_path)
                    st.success(f"Saved intent to {target_path}")
                except Exception as exc:
                    st.error(f"Failed to save intent: {exc}")

            saved_path = st.session_state.get("last_intent_save_path")
            commit_cols = st.columns([3, 1])
            commit_message = commit_cols[0].text_input(
                "Commit message",
                value=f"Add intent {generated_schema.get('name', 'intent')}",
                key="intent_commit_message",
            )
            if commit_cols[1].button(
                "Create Git Commit",
                key="intent_commit_btn",
                disabled=not saved_path,
            ):
                if saved_path:
                    success, output = git_commit([saved_path], commit_message)
                    if success:
                        st.success(f"Git commit created:\n{output}")
                    else:
                        st.error(f"Git commit failed: {output}")

            dry_run_result = dry_run_intent(generated_schema, tools)
            with st.expander("Dry-run analysis", expanded=False):
                status = dry_run_result.get("status")
                if status == "issues":
                    st.error("Issues detected during dry-run.")
                    for issue in dry_run_result.get("issues", []):
                        st.write(f"- {issue}")
                elif status == "ok":
                    st.success("All required tool inputs are satisfiable.")
                else:
                    st.info("Dry-run unavailable; ensure a schema and tools catalog are loaded.")
                st.json(dry_run_result.get("tasks", []))

            with st.expander("Raw LLM response", expanded=False):
                st.code(st.session_state.get("generated_intent_raw", ""), language="json")
        else:
            st.info("Generate an intent schema to view validation, diff, and export options.")

# ---------------------------
# Tab 4: Simulate
# ---------------------------
with mode[4]:
    st.subheader("Dry‑run the workflow")
    nodes = st.session_state.wf.as_nodes()
    edges = st.session_state.wf.as_edges()
    g = _mk_graph(nodes, edges)

    start_nodes = [n.id for n in nodes if n.type == "start"] or [nodes[0].id]
    start = start_nodes[0]
    st.write(f"Start at **{start}**")

    if "sim_ptr" not in st.session_state:
        st.session_state.sim_ptr = start

    def advance(ptr: str) -> Optional[str]:
        outs = list(g.successors(ptr))
        if not outs:
            return None
        if len(outs) == 1:
            return outs[0]
        # If decision, ask user which outbound edge to take using labels
        options = []
        for tgt in outs:
            lab = g.get_edge_data(ptr, tgt).get("label", "")
            options.append((tgt, lab or tgt))
        choice = st.radio("Choose path", options=[x[0] for x in options], format_func=lambda x: dict(options)[x], key=f"choice_{ptr}")
        return choice

    col1, col2 = st.columns([2, 1])
    with col1:
        mer = workflow_to_mermaid(st.session_state.wf)
        if _HAS_MERMAID:
            # FIX: unique key
            st_mermaid(mer, height=520, key="mermaid_preview_tab4")
        else:
            st.graphviz_chart(_mermaid_as_graphviz_dot(mer))
    with col2:
        st.write(f"**Current node:** `{st.session_state.sim_ptr}`")
        if st.button("Step →", key="step_btn"):
            nxt = advance(st.session_state.sim_ptr)
            if nxt is None:
                st.info("Reached a terminal node (no outgoing edges).")
            else:
                st.session_state.sim_ptr = nxt
        if st.button("Reset", key="reset_btn"):
            st.session_state.sim_ptr = start

# ---------------------------
# Footer
# ---------------------------
with st.expander("ℹ️ Help", expanded=False):
    st.markdown(
        """
        **How it works**
        - *Natural language*: Your text is converted into a Mermaid flowchart using an LLM (if configured) or a simple heuristic.
        - *Node/Edge editor*: Use the tables to add/edit nodes and edges, then update the diagram.
        - *Mermaid editor*: Power users can edit the Mermaid source directly.
        - *Simulate*: Step through the DAG to sanity‑check branching.

        **Environment variables (.env is loaded)**
        - `AZURE_OPENAI_ENDPOINT` – e.g., https://YOUR_RESOURCE.openai.azure.com
        - `AZURE_OPENAI_API_KEY`
        - `AZURE_OPENAI_DEPLOYMENT` – your GPT‑4o deployment name
        - `AZURE_OPENAI_API_VERSION` (optional, default: `2024-08-01-preview`)
        - or fallback:
          - `OPENAI_API_KEY`
          - `OPENAI_MODEL` (optional, default: `gpt-4o-mini`).

        **Export formats**
        - Mermaid (`.mmd`), JSON, YAML.

        **Tips**
        - Use *decision* node type for branches. Add edge labels like "yes"/"no".
        - Keep node IDs short and unique; labels can be verbose.
        - Choose *direction* in the sidebar for top‑down (`TD`) or left‑right (`LR`).
        """
    )
