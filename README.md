
# TopologicPy MCP Server

A [Model Context Protocol](https://modelcontextprotocol.io/) server that exposes [TopologicPy](https://topologic.app/topologicpy_doc/)'s spatial modeling capabilities to LLM agents like Claude Code, Claude Desktop, and other MCP clients.

## What This Enables

Ask Claude (or any MCP-equipped LLM) to build architectural models through natural language:

> "Create a 3-storey office building, 12m × 20m with 3.5m floor heights, then show me the adjacency graph and export to IFC"

The MCP server translates these requests into precise TopologicPy operations, maintaining a named object session across the conversation.

## Architecture

```
┌─────────────────────────────────────────────────┐
│              MCP Client (Claude Code, etc.)       │
│  Natural language ↔ tool calls                   │
└────────────────────┬────────────────────────────┘
                     │  MCP Protocol (stdio/SSE)
                     ▼
┌─────────────────────────────────────────────────┐
│            TopologicPy MCP Server                │
│  ┌───────────────────────────────────────────┐  │
│  │  Session Store (named topology objects)    │  │
│  │  • "building" → CellComplex               │  │
│  │  • "floor_0"  → Cell                      │  │
│  │  • "graph"    → Graph                     │  │
│  └───────────────────────────────────────────┘  │
│                                                  │
│  Tools: create, boolean, transform, query, I/O   │
│  Resources: session state, BREP strings          │
│  Prompts: building envelope, adjacency, grids    │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│                  TopologicPy                     │
│  Vertex → Edge → Wire → Face → Shell → Cell     │
│  CellComplex → Cluster → Graph → Dictionary     │
│  OpenCASCADE (BREP) • IfcOpenShell (IFC)        │
└─────────────────────────────────────────────────┘
```

## Available Tools (36 tools)

### Creation
| Tool | Description |
|------|-------------|
| `create_vertex` | Create a point at (x, y, z) |
| `create_edge` | Line segment between two vertices |
| `create_wire` | Polyline from ordered vertices |
| `create_face_by_wire` | Face from a closed wire |
| `create_face_rectangle` | Rectangular face |
| `create_face_circle` | Circular face (polygon approximation) |
| `create_cell_by_faces` | 3D solid from bounding faces |
| `create_cell_prism` | Box / rectangular prism |
| `create_cell_cylinder` | Cylindrical cell |
| `create_cellcomplex_by_cells` | Merged cell assembly (shared boundaries) |
| `create_cluster` | Unstructured collection |

### Boolean Operations
| Tool | Description |
|------|-------------|
| `boolean_union` | A ∪ B |
| `boolean_difference` | A − B |
| `boolean_intersect` | A ∩ B |
| `self_merge` | Resolve self-intersections |

### Transformations
| Tool | Description |
|------|-------------|
| `translate` | Move by vector |
| `rotate` | Rotate around axis |
| `scale` | Scale relative to origin |

### Query & Analysis
| Tool | Description |
|------|-------------|
| `list_topologies` | List all named objects in session |
| `query_topology` | Detailed info (counts, centroid, volume, area) |
| `get_vertices` | Extract vertex coordinates |
| `get_sub_topologies` | Extract and optionally store sub-elements |

### Graph Operations
| Tool | Description |
|------|-------------|
| `create_graph_from_topology` | Dual/adjacency graph |
| `graph_shortest_path` | Shortest path between vertices |

### Dictionary (Metadata)
| Tool | Description |
|------|-------------|
| `set_dictionary` | Attach key-value metadata |
| `get_dictionary` | Read attached metadata |

### Import / Export
| Tool | Description |
|------|-------------|
| `export_brep` | Export to BREP string/file |
| `import_brep` | Import from BREP string/file |
| `export_obj` | Export triangulated mesh (OBJ) |
| `export_ifc` | Export to IFC (BIM) |
| `import_ifc` | Import from IFC file |

### Session Management
| Tool | Description |
|------|-------------|
| `remove_topology` | Delete from session |
| `rename_topology` | Rename an object |
| `copy_topology` | Deep copy with new name |

## Installation

### With uv (recommended)
```bash
cd topologic-mcp-server
uv venv
source .venv/bin/activate
uv pip install -e .
```

### With pip
```bash
cd topologic-mcp-server
pip install -e .
```

## Usage

### With Claude Code
Add to your Claude Code MCP configuration (`~/.claude/claude_code_config.json`):

```json
{
  "mcpServers": {
    "topologic": {
      "command": "python",
      "args": ["-m", "topologic_mcp"],
      "cwd": "/path/to/topologic-mcp-server"
    }
  }
}
```

Or using uv directly:
```json
{
  "mcpServers": {
    "topologic": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/topologic-mcp-server", "topologic-mcp"]
    }
  }
}
```

### With Claude Desktop
Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "topologic": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/topologic-mcp-server", "topologic-mcp"]
    }
  }
}
```

### Standalone (stdio transport)
```bash
python -m topologic_mcp
```

### With MCP Inspector (for testing)
```bash
mcp dev src/topologic_mcp/server.py
```

## Example Conversations

### Create a simple building
```
User: Create a 3-storey building, 10m × 15m, 3m floor height

Claude: [calls create_vertex, create_cell_prism × 3, translate × 2,
         create_cellcomplex_by_cells, query_topology]

Result: CellComplex "building" with 3 cells, 16 faces, 33 edges, 20 vertices
```

### Analyze spatial adjacency
```
User: Show me which floors share faces in the building

Claude: [calls create_graph_from_topology with via_shared_faces=True,
         get_sub_topologies to list cells, query each cell]

Result: Graph with 3 vertices (one per floor) and 2 edges
        (floor_0↔floor_1, floor_1↔floor_2)
```

### Boolean operations
```
User: Cut a 2m diameter hole through the middle of floor_1

Claude: [calls create_cell_cylinder for the hole, translate to position,
         boolean_difference to subtract from floor_1]

Result: Updated floor_1 with cylindrical void
```

## License

GPL-3.0-or-later (matching TopologicPy's license)

## Contributing

This server wraps TopologicPy's pure-Python API. To add new tools:
1. Add a `@mcp.tool()` decorated function in `server.py`
2. Follow the naming convention: `verb_noun` (e.g., `create_vertex`, `export_brep`)
3. Always accept `ctx: Context` as the first parameter
4. Use the `TopologyStore` from `ctx.request_context.lifespan_context`
5. Return descriptive strings (the LLM reads these)
