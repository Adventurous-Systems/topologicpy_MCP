# TopologicPy MCP Server — Token Optimization Plan

## Current State Analysis

### Token Budget Breakdown (per request, 30 tools)

| Component | Est. Tokens | % of Request | Sent Every Call? |
|---|---|---|---|
| Tool definitions (30 tools, schemas + descriptions) | ~3,200 | 60-85% | YES |
| Server description | ~80 | 1-2% | YES |
| Prompt templates (3) | ~400 | 5-10% | If enumerated |
| Resource URIs (2) | ~100 | 1-3% | If enumerated |
| User message | ~20-100 | 2-5% | YES |
| LLM reasoning + tool call | ~50-200 | 3-10% | YES |
| Tool response | ~20-15,000+ | 1-80% | YES |

### Key Problems Identified

1. **Tool Definition Tax**: 30 tool schemas (~3,200 tokens) sent with EVERY request, even for simple queries
2. **BREP Token Bomb**: `export_brep` without `file_path` returns raw BREP strings (5KB-500KB+) as tool response, consuming 1,500-150,000+ tokens
3. **Cumulative History**: Multi-turn conversations re-send all prior turns, growing linearly
4. **No Response Compression**: JSON responses from `query_topology` and `list_topologies` use verbose formatting
5. **Redundant Descriptions**: Tool descriptions repeat parameter info already in the JSON schema

---

## Optimization Strategies

### Phase 1: Quick Wins (No Architecture Changes)

#### 1.1 Compress Tool Descriptions ← HIGHEST IMPACT
**Savings: ~800-1,200 tokens per request (25-35%)**

Current tool descriptions include verbose Args/Returns sections that duplicate the JSON schema. Trim to essential behavior description only.

**Before (create_vertex, 310 chars):**
```
Create a Vertex (point) at the given coordinates and store it with the given name.

Args:
    name: Unique name to reference this vertex later.
    x: X coordinate. Default 0.
    y: Y coordinate. Default 0.
    z: Z coordinate. Default 0.

Returns:
    Confirmation with vertex details.
```

**After (create_vertex, 85 chars):**
```
Create a Vertex (point) at (x,y,z) and store it by name.
```

The Args/Returns sections are redundant — the JSON schema already declares parameter names, types, and defaults. LLMs read schemas natively.

**Action**: Rewrite all 30 tool docstrings to single-line descriptions. Remove Args/Returns sections entirely.

#### 1.2 Force File-Based BREP Export ← PREVENTS TOKEN BOMBS
**Savings: 1,500-150,000+ tokens per BREP export**

Currently `export_brep` returns the raw BREP string if no `file_path` is given. This is extremely wasteful — a simple box is ~5KB, a building is 50-500KB.

**Action**:
- Make `file_path` required in `export_brep`
- Return only a confirmation message with file path and size
- Same for `import_brep`: accept file_path only, remove `brep_string` parameter
- Remove the `topologic://topology/{name}/brep` resource (or add a max-size guard)

#### 1.3 Compact JSON Responses
**Savings: ~20-40% on query/list responses**

**Action**:
- Use `json.dumps(data, separators=(',', ':'))` instead of `indent=2`
- Truncate coordinate precision to 2 decimal places
- In `list_topologies`, return compact format: `"room:Cell, floor_0:Cell"` instead of full JSON array

#### 1.4 Shorter Server Description
**Savings: ~40 tokens per request**

**Before (310 chars):**
```
MCP server for TopologicPy — a spatial modeling library for creating hierarchical,
topological representations of architectural spaces, buildings, and artefacts using
non-manifold topology (NMT). Supports Vertex, Edge, Wire, Face, Shell, Cell,
CellComplex, Cluster, Graph, Dictionary, and Boolean operations, plus IFC/OBJ/BREP
import/export.
```

**After (120 chars):**
```
TopologicPy spatial modeling: create/query/transform/export 3D topological models
(Vertex→Edge→Wire→Face→Shell→Cell→CellComplex).
```

---

### Phase 2: Tool Organization (Moderate Changes)

#### 2.1 Tool Grouping / Namespacing
**Savings: Better LLM tool selection accuracy, fewer wasted calls**

Group tools with prefixed names to help LLMs navigate:
- `topo.create_vertex`, `topo.create_edge`, etc.
- `topo.bool_union`, `topo.bool_difference`
- `topo.export_brep`, `topo.export_obj`

This doesn't reduce tokens but improves LLM accuracy (fewer retries = fewer tokens).

#### 2.2 Merge Similar Tools
**Savings: ~500-800 tokens from tool definitions**

Merge tools that share the same pattern:

| Current (separate tools) | Merged tool |
|---|---|
| `create_face_rectangle` + `create_face_circle` | `create_face(shape="rectangle"\|"circle", ...)` |
| `boolean_union` + `boolean_difference` + `boolean_intersect` | `boolean(operation="union"\|"difference"\|"intersect", ...)` |
| `export_brep` + `export_obj` + `export_ifc` | `export(format="brep"\|"obj"\|"ifc", ...)` |
| `remove_topology` + `rename_topology` + `copy_topology` | `manage_topology(action="remove"\|"rename"\|"copy", ...)` |

This reduces 30 tools → ~18 tools, saving ~40% of definition overhead.

#### 2.3 Lazy Tool Loading (MCP Protocol Level)
**Savings: Only load tools when needed**

The MCP protocol doesn't natively support this, but you can implement it at the client level:

**Approach A — Tool Categories**: Register a single "meta-tool" that returns available tools for a category:
```python
@mcp.tool()
def get_tools(category: str) -> str:
    """List available operations. Categories: create, boolean, transform, query, graph, io, session"""
```

**Approach B — Two-Server Architecture**: Split into a "router" MCP server (5 tools: create, modify, query, export, manage) that internally dispatches to the full API.

---

### Phase 3: Response Optimization (Architecture Changes)

#### 3.1 Pagination for Large Results
**Savings: Prevents unbounded responses**

For `get_vertices` on complex models (100+ vertices), return paginated results:
```python
def get_vertices(ctx, topology_name, offset=0, limit=20) -> str:
    # Returns max 20 vertices per call
```

#### 3.2 Summary vs Detail Modes
**Savings: 50-90% on query responses**

Add a `detail` parameter to query tools:
```python
def query_topology(ctx, topology_name, detail="summary") -> str:
    # summary: type, vertex count, volume (1 line)
    # full: all counts, centroid, bbox, etc.
```

#### 3.3 Session State Compression
**Savings: Reduces context growth in multi-turn sessions**

Instead of the LLM tracking session state in conversation history, provide a compact session summary tool:
```python
def session_state() -> str:
    """Returns compact session state: 'room:Cell(v=8,e=12), building:CC(cells=3)'"""
```

This lets the LLM "forget" intermediate steps and rely on the server for current state.

---

### Phase 4: Platform-Specific Optimizations

#### 4.1 Claude Code Optimizations
- Use `claude-haiku-4-5-20251001` for simple tool dispatch (3x cheaper than Sonnet)
- Leverage prompt caching: tool definitions are static and cache well
- Use Resources instead of tool calls for read-only state queries

#### 4.2 ChatGPT/OpenAI Optimizations
- Use `gpt-4o-mini` for tool routing, `gpt-4o` only for complex planning
- OpenAI's function calling is slightly more token-efficient than system prompt tools
- Consider structured outputs for deterministic tool parameter generation

#### 4.3 Ollama/Local Model Optimizations
- **Critical**: Local models have 8K-32K context windows
- Must reduce to ≤10 tools for 8K models
- Use the merged tool approach (Phase 2.2) as mandatory
- Consider a "lite" server profile with only 8 essential tools:
  1. `create_geometry` (vertex/edge/face/cell/prism)
  2. `boolean` (union/diff/intersect)
  3. `transform` (translate/rotate/scale)
  4. `query` (list/query/vertices)
  5. `graph` (create/shortest_path)
  6. `metadata` (get/set dictionary)
  7. `export` (brep/obj/ifc)
  8. `session` (remove/rename/copy)

---

## Implementation Priority & Impact

| Optimization | Effort | Token Savings | Priority |
|---|---|---|---|
| 1.1 Compress descriptions | 1 hour | 25-35% per request | **P0** |
| 1.2 Force file-based BREP | 30 min | Prevents 150K+ bombs | **P0** |
| 1.3 Compact JSON | 30 min | 20-40% on responses | **P1** |
| 1.4 Shorter server desc | 5 min | ~40 tokens/request | **P1** |
| 2.2 Merge similar tools | 2-3 hours | 30-40% definitions | **P1** |
| 3.2 Summary/detail modes | 1 hour | 50-90% on queries | **P1** |
| 3.1 Pagination | 1 hour | Prevents unbounded | **P2** |
| 3.3 Session compression | 1-2 hours | Reduces history growth | **P2** |
| 4.3 Lite server profile | 2-3 hours | Required for Ollama 8K | **P2** |
| 2.3 Lazy tool loading | 3-5 hours | 60-80% definitions | **P3** |

---

## Expected Results

### Before Optimization (building_envelope workflow, 9 steps)
| Platform | Tokens/Workflow | Cost |
|---|---|---|
| Claude Sonnet | ~45,000 | ~$0.18 |
| GPT-4o | ~42,000 | ~$0.15 |
| Ollama 8K | OVERFLOW | N/A |
| Ollama 32K | ~42,000 | FREE |

### After Phase 1+2 Optimization (same workflow)
| Platform | Tokens/Workflow | Cost | Savings |
|---|---|---|---|
| Claude Sonnet | ~22,000 | ~$0.09 | 51% |
| GPT-4o | ~20,000 | ~$0.07 | 53% |
| Ollama 8K | ~7,500 (lite) | FREE | Fits! |
| Ollama 32K | ~20,000 | FREE | 52% |

---

## Running the Benchmarks

### Static Analysis (no API keys needed)
```bash
cd /path/to/topologicpy_MCP
python benchmarks/token_benchmark.py
```

### Live Benchmark (requires API keys)
```bash
# Claude only
ANTHROPIC_API_KEY=sk-... python benchmarks/live_benchmark.py --provider claude

# OpenAI only
OPENAI_API_KEY=sk-... python benchmarks/live_benchmark.py --provider openai

# Ollama (must be running locally)
python benchmarks/live_benchmark.py --provider ollama --model llama3.1:8b

# All providers
ANTHROPIC_API_KEY=sk-... OPENAI_API_KEY=sk-... python benchmarks/live_benchmark.py --all
```

### Interpreting Results
- **Tool definition overhead %**: Should be <30% after optimization (currently 60-85%)
- **Context usage %**: Should be <5% per step for Ollama 8K
- **Cost per workflow**: Target <$0.10 for building_envelope on Claude/GPT-4o
