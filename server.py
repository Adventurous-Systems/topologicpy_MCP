"""
TopologicPy MCP Server
======================
A Model Context Protocol server that exposes TopologicPy's spatial modeling
capabilities to LLM agents (Claude Code, etc.).

Architecture:
- Tools: Create, query, transform, and export topological entities
- Resources: Expose the current model state, entity metadata, BREP strings
- Prompts: Common workflows (building envelope, space analysis, etc.)

The server maintains an in-memory session with named topology objects,
allowing incremental model building through natural language.
"""

from __future__ import annotations

import json
import uuid
import logging
from typing import Any
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from mcp.server.fastmcp import FastMCP, Context

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Session State — persists named topologies across tool calls
# ---------------------------------------------------------------------------

@dataclass
class TopologyStore:
    """In-memory store of named topology objects for the current session."""
    objects: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, dict] = field(default_factory=dict)

    def put(self, name: str, topo: Any, meta: dict | None = None) -> str:
        self.objects[name] = topo
        self.metadata[name] = meta or {}
        return name

    def get(self, name: str) -> Any:
        if name not in self.objects:
            raise KeyError(f"No topology named '{name}'. Available: {list(self.objects.keys())}")
        return self.objects[name]

    def remove(self, name: str) -> bool:
        if name in self.objects:
            del self.objects[name]
            self.metadata.pop(name, None)
            return True
        return False

    def list_all(self) -> list[dict]:
        from topologicpy.Topology import Topology
        result = []
        for name, topo in self.objects.items():
            entry = {
                "name": name,
                "type": Topology.TypeAsString(topo),
                **self.metadata.get(name, {}),
            }
            result.append(entry)
        return result


# ---------------------------------------------------------------------------
# Lifespan — initialize TopologicPy and the store
# ---------------------------------------------------------------------------

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[TopologyStore]:
    """Initialize the topology store on startup."""
    logger.info("TopologicPy MCP server starting...")
    store = TopologyStore()
    try:
        yield store
    finally:
        logger.info("TopologicPy MCP server shutting down.")


# ---------------------------------------------------------------------------
# Server Definition
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "topologic",
    instructions=(
        "TopologicPy spatial modeling: create/query/transform/export 3D "
        "topological models (Vertex, Edge, Wire, Face, Shell, Cell, "
        "CellComplex, Cluster, Graph). Supports boolean ops and IFC/OBJ/BREP I/O."
    ),
    lifespan=app_lifespan,
    dependencies=["topologicpy"],
)


# ═══════════════════════════════════════════════════════════════════════════
# TOOLS — Topology Creation
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
def create_vertex(
    ctx: Context,
    name: str,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
) -> str:
    """Create a Vertex (point) at (x,y,z) and store it by name."""
    from topologicpy.Vertex import Vertex
    store: TopologyStore = ctx.request_context.lifespan_context
    v = Vertex.ByCoordinates(x, y, z)
    store.put(name, v, {"x": x, "y": y, "z": z})
    return f"Created Vertex '{name}' at ({x}, {y}, {z})"


@mcp.tool()
def create_edge(
    ctx: Context,
    name: str,
    start_vertex: str,
    end_vertex: str,
    tolerance: float = 0.0001,
) -> str:
    """Create an Edge (line segment) between two named vertices."""
    from topologicpy.Edge import Edge
    store: TopologyStore = ctx.request_context.lifespan_context
    sv = store.get(start_vertex)
    ev = store.get(end_vertex)
    e = Edge.ByVertices([sv, ev], tolerance=tolerance)
    store.put(name, e, {"start": start_vertex, "end": end_vertex})
    return f"Created Edge '{name}' from '{start_vertex}' to '{end_vertex}'"


@mcp.tool()
def create_wire(
    ctx: Context,
    name: str,
    vertex_names: list[str],
    close: bool = True,
    tolerance: float = 0.0001,
) -> str:
    """Create a Wire (polyline) from an ordered list of named vertices. Set close=True to connect last to first."""
    from topologicpy.Wire import Wire
    store: TopologyStore = ctx.request_context.lifespan_context
    vertices = [store.get(vn) for vn in vertex_names]
    w = Wire.ByVertices(vertices, close=close, tolerance=tolerance)
    store.put(name, w, {"vertices": vertex_names, "closed": close})
    return f"Created Wire '{name}' from {len(vertex_names)} vertices (closed={close})"


@mcp.tool()
def create_face_by_wire(
    ctx: Context,
    name: str,
    wire_name: str,
    tolerance: float = 0.0001,
) -> str:
    """Create a Face from a named closed Wire (external boundary)."""
    from topologicpy.Face import Face
    store: TopologyStore = ctx.request_context.lifespan_context
    wire = store.get(wire_name)
    f = Face.ByWire(wire, tolerance=tolerance)
    store.put(name, f, {"boundary_wire": wire_name})
    return f"Created Face '{name}' from Wire '{wire_name}'"


@mcp.tool()
def create_face_shape(
    ctx: Context,
    name: str,
    shape: str,
    origin_name: str | None = None,
    width: float = 1.0,
    length: float = 1.0,
    radius: float = 0.5,
    sides: int = 32,
    direction: list[float] | None = None,
    tolerance: float = 0.0001,
) -> str:
    """Create a Face. shape: "rectangle"|"circle". Rectangle uses width/length, circle uses radius/sides."""
    from topologicpy.Face import Face
    from topologicpy.Vertex import Vertex
    store: TopologyStore = ctx.request_context.lifespan_context
    origin = store.get(origin_name) if origin_name else Vertex.ByCoordinates(0, 0, 0)
    direction = direction or [0, 0, 1]
    if shape == "rectangle":
        f = Face.Rectangle(
            origin=origin, width=width, length=length,
            direction=direction, tolerance=tolerance,
        )
        store.put(name, f, {"shape": "rectangle", "width": width, "length": length, "direction": direction})
        return f"Created rectangular Face '{name}' ({width} x {length})"
    elif shape == "circle":
        f = Face.Circle(
            origin=origin, radius=radius, sides=sides,
            direction=direction, tolerance=tolerance,
        )
        store.put(name, f, {"shape": "circle", "radius": radius, "sides": sides})
        return f"Created circular Face '{name}' (radius={radius}, sides={sides})"
    else:
        return f"Invalid shape '{shape}'. Use 'rectangle' or 'circle'."


@mcp.tool()
def create_cell_by_faces(
    ctx: Context,
    name: str,
    face_names: list[str],
    tolerance: float = 0.0001,
) -> str:
    """Create a Cell (3D solid) from named Faces. Faces must form a closed watertight shell."""
    from topologicpy.Cell import Cell
    store: TopologyStore = ctx.request_context.lifespan_context
    faces = [store.get(fn) for fn in face_names]
    c = Cell.ByFaces(faces, tolerance=tolerance)
    store.put(name, c, {"faces": face_names})
    return f"Created Cell '{name}' from {len(face_names)} faces"


@mcp.tool()
def create_cell_shape(
    ctx: Context,
    name: str,
    shape: str,
    origin_name: str | None = None,
    width: float = 1.0,
    length: float = 1.0,
    height: float = 1.0,
    radius: float = 0.5,
    sides: int = 32,
    placement: str = "center",
    tolerance: float = 0.0001,
) -> str:
    """Create a Cell. shape: "prism"|"cylinder". Prism uses width/length/height, cylinder uses radius/height/sides."""
    from topologicpy.Cell import Cell
    from topologicpy.Vertex import Vertex
    store: TopologyStore = ctx.request_context.lifespan_context
    origin = store.get(origin_name) if origin_name else Vertex.ByCoordinates(0, 0, 0)
    if shape == "prism":
        c = Cell.Prism(
            origin=origin, width=width, length=length, height=height,
            placement=placement, tolerance=tolerance,
        )
        store.put(name, c, {"shape": "prism", "width": width, "length": length, "height": height})
        return f"Created prism Cell '{name}' ({width} x {length} x {height})"
    elif shape == "cylinder":
        c = Cell.Cylinder(
            origin=origin, radius=radius, height=height, uSides=sides,
            placement=placement, tolerance=tolerance,
        )
        store.put(name, c, {"shape": "cylinder", "radius": radius, "height": height, "sides": sides})
        return f"Created cylindrical Cell '{name}' (r={radius}, h={height})"
    else:
        return f"Invalid shape '{shape}'. Use 'prism' or 'cylinder'."


@mcp.tool()
def create_cellcomplex_by_cells(
    ctx: Context,
    name: str,
    cell_names: list[str],
    tolerance: float = 0.0001,
) -> str:
    """Create a CellComplex by merging named Cells, sharing faces/edges/vertices at intersections."""
    from topologicpy.CellComplex import CellComplex
    store: TopologyStore = ctx.request_context.lifespan_context
    cells = [store.get(cn) for cn in cell_names]
    cc = CellComplex.ByCells(cells, tolerance=tolerance)
    store.put(name, cc, {"cells": cell_names})
    return f"Created CellComplex '{name}' from {len(cell_names)} cells"


@mcp.tool()
def create_cluster(
    ctx: Context,
    name: str,
    topology_names: list[str],
) -> str:
    """Create a Cluster (unstructured collection) from named topologies."""
    from topologicpy.Cluster import Cluster
    store: TopologyStore = ctx.request_context.lifespan_context
    topos = [store.get(tn) for tn in topology_names]
    cl = Cluster.ByTopologies(topos)
    store.put(name, cl, {"members": topology_names})
    return f"Created Cluster '{name}' with {len(topology_names)} members"


# ═══════════════════════════════════════════════════════════════════════════
# TOOLS — Boolean Operations
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
def boolean(
    ctx: Context,
    name: str,
    operation: str,
    topology_a: str,
    topology_b: str | None = None,
    tolerance: float = 0.0001,
) -> str:
    """Boolean op: "union"|"difference"|"intersect"|"self_merge". topology_b not needed for self_merge."""
    from topologicpy.Topology import Topology
    store: TopologyStore = ctx.request_context.lifespan_context
    a = store.get(topology_a)
    if operation == "self_merge":
        result = Topology.SelfMerge(a, tolerance=tolerance)
        store.put(name, result, {"operation": "self_merge", "source": topology_a})
        return f"Created '{name}' = SelfMerge('{topology_a}') → {Topology.TypeAsString(result)}"
    if topology_b is None:
        return "topology_b is required for union/difference/intersect operations."
    b = store.get(topology_b)
    ops = {
        "union": Topology.Union,
        "difference": Topology.Difference,
        "intersect": Topology.Intersect,
    }
    if operation not in ops:
        return f"Invalid operation '{operation}'. Use 'union', 'difference', 'intersect', or 'self_merge'."
    result = ops[operation](a, b, tolerance=tolerance)
    store.put(name, result, {"operation": operation, "a": topology_a, "b": topology_b})
    return f"Created '{name}' = {operation.title()}('{topology_a}', '{topology_b}') → {Topology.TypeAsString(result)}"


# ═══════════════════════════════════════════════════════════════════════════
# TOOLS — Transformations
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
def transform(
    ctx: Context,
    name: str,
    topology_name: str,
    operation: str,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    origin_name: str | None = None,
    axis: list[float] | None = None,
    angle: float = 0.0,
    x_factor: float = 1.0,
    y_factor: float = 1.0,
    z_factor: float = 1.0,
) -> str:
    """Transform topology. operation: "translate"|"rotate"|"scale". Translate uses x/y/z, rotate uses origin/axis/angle, scale uses origin/x_factor/y_factor/z_factor."""
    from topologicpy.Topology import Topology
    from topologicpy.Vertex import Vertex
    store: TopologyStore = ctx.request_context.lifespan_context
    topo = store.get(topology_name)
    if operation == "translate":
        result = Topology.Translate(topo, x, y, z)
        store.put(name, result, {"operation": "translate", "source": topology_name, "vector": [x, y, z]})
        return f"Created '{name}' = Translate('{topology_name}', [{x}, {y}, {z}])"
    elif operation == "rotate":
        origin = store.get(origin_name) if origin_name else Vertex.ByCoordinates(0, 0, 0)
        axis = axis or [0, 0, 1]
        result = Topology.Rotate(topo, origin=origin, axis=axis, angle=angle)
        store.put(name, result, {"operation": "rotate", "source": topology_name, "angle": angle})
        return f"Created '{name}' = Rotate('{topology_name}', angle={angle}°, axis={axis})"
    elif operation == "scale":
        origin = store.get(origin_name) if origin_name else Vertex.ByCoordinates(0, 0, 0)
        result = Topology.Scale(topo, origin=origin, x=x_factor, y=y_factor, z=z_factor)
        store.put(name, result, {"operation": "scale", "source": topology_name})
        return f"Created '{name}' = Scale('{topology_name}', factors=[{x_factor}, {y_factor}, {z_factor}])"
    else:
        return f"Invalid operation '{operation}'. Use 'translate', 'rotate', or 'scale'."


# ═══════════════════════════════════════════════════════════════════════════
# TOOLS — Query & Analysis
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
def list_topologies(ctx: Context) -> str:
    """List all named topologies in the session (name, type, metadata)."""
    store: TopologyStore = ctx.request_context.lifespan_context
    items = store.list_all()
    if not items:
        return "Session empty."
    return json.dumps(items, separators=(',', ':'))


@mcp.tool()
def query_topology(
    ctx: Context,
    topology_name: str,
    detail: str = "summary",
) -> str:
    """Query a named topology. detail: "summary" (one-line type+counts) or "full" (type, counts, centroid, bounding box, volume/area)."""
    from topologicpy.Topology import Topology
    from topologicpy.Vertex import Vertex
    from topologicpy.Cell import Cell
    from topologicpy.Face import Face
    store: TopologyStore = ctx.request_context.lifespan_context
    topo = store.get(topology_name)
    topo_type = Topology.TypeAsString(topo)

    vertices = Topology.Vertices(topo) or []
    edges = Topology.Edges(topo) or []
    faces = Topology.Faces(topo) or []
    cells = Topology.Cells(topo) or []

    if detail == "summary":
        return f"{topology_name}: {topo_type} (V:{len(vertices)} E:{len(edges)} F:{len(faces)} C:{len(cells)})"

    info: dict[str, Any] = {
        "name": topology_name,
        "type": topo_type,
        "counts": {
            "vertices": len(vertices),
            "edges": len(edges),
            "faces": len(faces),
            "cells": len(cells),
        },
    }

    centroid = Topology.Centroid(topo)
    if centroid:
        info["centroid"] = {
            "x": round(Vertex.X(centroid), 2),
            "y": round(Vertex.Y(centroid), 2),
            "z": round(Vertex.Z(centroid), 2),
        }

    try:
        bb = Topology.BoundingBox(topo)
        if bb:
            info["bounding_box"] = str(bb)
    except Exception:
        pass

    if topo_type == "Cell":
        try:
            vol = Cell.Volume(topo)
            info["volume"] = round(vol, 2)
        except Exception:
            pass

    if topo_type == "Face":
        try:
            area = Face.Area(topo)
            info["area"] = round(area, 2)
        except Exception:
            pass

    return json.dumps(info, separators=(',', ':'))


@mcp.tool()
def get_vertices(
    ctx: Context,
    topology_name: str,
    mantissa: int = 6,
) -> str:
    """Get all vertex coordinates of a named topology as [[x,y,z], ...]."""
    from topologicpy.Topology import Topology
    from topologicpy.Vertex import Vertex
    store: TopologyStore = ctx.request_context.lifespan_context
    topo = store.get(topology_name)
    vertices = Topology.Vertices(topo) or []
    coords = [
        [round(Vertex.X(v), mantissa), round(Vertex.Y(v), mantissa), round(Vertex.Z(v), mantissa)]
        for v in vertices
    ]
    return json.dumps(coords)


@mcp.tool()
def get_sub_topologies(
    ctx: Context,
    topology_name: str,
    sub_type: str,
    store_with_prefix: str | None = None,
) -> str:
    """Extract sub-topologies from a topology. sub_type: "vertex"|"edge"|"wire"|"face"|"shell"|"cell". Optionally store with prefix."""
    from topologicpy.Topology import Topology
    store: TopologyStore = ctx.request_context.lifespan_context
    topo = store.get(topology_name)

    extractors = {
        "vertex": Topology.Vertices,
        "edge": Topology.Edges,
        "wire": Topology.Wires,
        "face": Topology.Faces,
        "shell": Topology.Shells,
        "cell": Topology.Cells,
    }

    if sub_type.lower() not in extractors:
        return f"Invalid sub_type '{sub_type}'. Choose from: {list(extractors.keys())}"

    subs = extractors[sub_type.lower()](topo) or []
    names = []

    if store_with_prefix:
        for i, sub in enumerate(subs):
            sub_name = f"{store_with_prefix}_{i}"
            store.put(sub_name, sub, {"parent": topology_name, "index": i, "sub_type": sub_type})
            names.append(sub_name)

    result = {"count": len(subs), "sub_type": sub_type}
    if names:
        result["stored_as"] = names
    return json.dumps(result, separators=(',', ':'))


# ═══════════════════════════════════════════════════════════════════════════
# TOOLS — Graph Operations
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
def create_graph_from_topology(
    ctx: Context,
    name: str,
    topology_name: str,
    direct: bool = True,
    via_shared_topologies: bool = False,
    via_shared_apertures: bool = False,
    to_exterior_topologies: bool = False,
    to_exterior_apertures: bool = False,
    tolerance: float = 0.0001,
) -> str:
    """Create a dual/adjacency Graph from a topology (e.g. cell adjacency in a CellComplex). Configure connectivity via shared topologies/apertures."""
    from topologicpy.Graph import Graph
    store: TopologyStore = ctx.request_context.lifespan_context
    topo = store.get(topology_name)
    g = Graph.ByTopology(
        topo,
        direct=direct,
        viaSharedTopologies=via_shared_topologies,
        viaSharedApertures=via_shared_apertures,
        toExteriorTopologies=to_exterior_topologies,
        toExteriorApertures=to_exterior_apertures,
        tolerance=tolerance,
    )
    verts = Graph.Vertices(g) or []
    edges = Graph.Edges(g) or []
    store.put(name, g, {"source": topology_name, "vertices": len(verts), "edges": len(edges)})
    return f"Created Graph '{name}' from '{topology_name}' ({len(verts)} vertices, {len(edges)} edges)"


@mcp.tool()
def graph_shortest_path(
    ctx: Context,
    name: str,
    graph_name: str,
    start_vertex_name: str,
    end_vertex_name: str,
    tolerance: float = 0.0001,
) -> str:
    """Find the shortest path between two named vertices in a graph."""
    from topologicpy.Graph import Graph
    from topologicpy.Topology import Topology
    store: TopologyStore = ctx.request_context.lifespan_context
    g = store.get(graph_name)
    sv = store.get(start_vertex_name)
    ev = store.get(end_vertex_name)
    path = Graph.ShortestPath(g, sv, ev, tolerance=tolerance)
    if path:
        verts = Topology.Vertices(path) or []
        store.put(name, path, {"graph": graph_name, "path_vertices": len(verts)})
        return f"Found shortest path '{name}' with {len(verts)} vertices"
    return "No path found between the specified vertices."


# ═══════════════════════════════════════════════════════════════════════════
# TOOLS — Dictionary (Metadata) Operations
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
def dictionary(
    ctx: Context,
    topology_name: str,
    action: str = "get",
    data: dict[str, Any] | None = None,
) -> str:
    """Get or set metadata. action: "get"|"set". data required for set."""
    from topologicpy.Topology import Topology
    from topologicpy.Dictionary import Dictionary
    store: TopologyStore = ctx.request_context.lifespan_context
    topo = store.get(topology_name)
    if action == "set":
        if not data:
            return "data is required for action='set'."
        keys = list(data.keys())
        values = list(data.values())
        d = Dictionary.ByKeysValues(keys, values)
        topo = Topology.SetDictionary(topo, d)
        store.objects[topology_name] = topo
        return f"Set dictionary on '{topology_name}' with keys: {keys}"
    elif action == "get":
        d = Topology.Dictionary(topo)
        if d is None:
            return f"No dictionary attached to '{topology_name}'"
        keys = Dictionary.Keys(d) or []
        values = Dictionary.Values(d) or []
        return json.dumps(dict(zip(keys, values)), separators=(',', ':'), default=str)
    else:
        return f"Invalid action '{action}'. Use 'get' or 'set'."


# ═══════════════════════════════════════════════════════════════════════════
# TOOLS — Import / Export (merged: brep+obj+ifc)
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
def export_topology(
    ctx: Context,
    topology_name: str,
    file_path: str,
    format: str,
    transpose: bool = False,
    building_name: str = "Default Building",
    tolerance: float = 0.0001,
) -> str:
    """Export topology to file. format: "brep"|"obj"|"ifc"."""
    from topologicpy.Topology import Topology
    store: TopologyStore = ctx.request_context.lifespan_context
    topo = store.get(topology_name)
    if format == "brep":
        brep_string = Topology.BREPString(topo)
        with open(file_path, "w") as f:
            f.write(brep_string)
        return f"Exported '{topology_name}' BREP to {file_path} ({len(brep_string)} chars)"
    elif format == "obj":
        Topology.ExportToOBJ(topo, path=file_path, transposeAxes=transpose, tolerance=tolerance)
        return f"Exported '{topology_name}' to OBJ: {file_path}"
    elif format == "ifc":
        Topology.ExportToIFC(topo, path=file_path, buildingName=building_name)
        return f"Exported '{topology_name}' to IFC: {file_path}"
    else:
        return f"Invalid format '{format}'. Use 'brep', 'obj', or 'ifc'."


@mcp.tool()
def import_topology(
    ctx: Context,
    name: str,
    file_path: str,
    format: str,
    tolerance: float = 0.0001,
) -> str:
    """Import topology from file. format: "brep"|"ifc"."""
    from topologicpy.Topology import Topology
    store: TopologyStore = ctx.request_context.lifespan_context
    if format == "brep":
        with open(file_path, "r") as f:
            brep_string = f.read()
        topo = Topology.ByBREPString(brep_string)
        store.put(name, topo, {"source": "brep_import", "file": file_path})
        return f"Imported '{name}' as {Topology.TypeAsString(topo)} from {file_path}"
    elif format == "ifc":
        topo = Topology.ByIFCFile(file_path, tolerance=tolerance)
        store.put(name, topo, {"source": "ifc_import", "file": file_path})
        return f"Imported '{name}' from IFC as {Topology.TypeAsString(topo)}"
    else:
        return f"Invalid format '{format}'. Use 'brep' or 'ifc'."


# ═══════════════════════════════════════════════════════════════════════════
# TOOLS — Session Management (merged: remove+rename+copy)
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
def manage_topology(
    ctx: Context,
    action: str,
    topology_name: str,
    new_name: str | None = None,
) -> str:
    """Manage session. action: "remove"|"rename"|"copy". new_name required for rename/copy."""
    from topologicpy.Topology import Topology
    store: TopologyStore = ctx.request_context.lifespan_context
    if action == "remove":
        if store.remove(topology_name):
            return f"Removed '{topology_name}' from store."
        return f"'{topology_name}' not found in store."
    elif action == "rename":
        if not new_name:
            return "new_name is required for action='rename'."
        topo = store.get(topology_name)
        meta = store.metadata.get(topology_name, {})
        store.remove(topology_name)
        store.put(new_name, topo, meta)
        return f"Renamed '{topology_name}' → '{new_name}'"
    elif action == "copy":
        if not new_name:
            return "new_name is required for action='copy'."
        topo = store.get(topology_name)
        copy = Topology.Copy(topo)
        meta = store.metadata.get(topology_name, {}).copy()
        meta["copied_from"] = topology_name
        store.put(new_name, copy, meta)
        return f"Copied '{topology_name}' → '{new_name}'"
    else:
        return f"Invalid action '{action}'. Use 'remove', 'rename', or 'copy'."


# ═══════════════════════════════════════════════════════════════════════════
# RESOURCES — Model State
# ═══════════════════════════════════════════════════════════════════════════

@mcp.resource("topologic://session/summary")
def session_summary(ctx: Context) -> str:
    """Summary of all topologies in the current session."""
    store: TopologyStore = ctx.request_context.lifespan_context
    items = store.list_all()
    if not items:
        return "Session empty."
    return json.dumps(items, separators=(',', ':'))


@mcp.resource("topologic://topology/{name}/brep")
def topology_brep_resource(name: str, ctx: Context) -> str:
    """Get BREP info for a named topology. Returns size only — use export_brep tool to save full BREP to file."""
    from topologicpy.Topology import Topology
    store: TopologyStore = ctx.request_context.lifespan_context
    topo = store.get(name)
    brep_string = Topology.BREPString(topo)
    return f"BREP for '{name}': {len(brep_string)} chars. Use export_brep tool to save to file."


# ═══════════════════════════════════════════════════════════════════════════
# PROMPTS — Common Workflows
# ═══════════════════════════════════════════════════════════════════════════

@mcp.prompt()
def building_envelope(
    width: float = 10.0,
    length: float = 15.0,
    height: float = 3.0,
    floors: int = 3,
) -> str:
    """Generate a prompt to create a multi-storey building envelope as a CellComplex.

    Args:
        width: Building width in meters.
        length: Building length in meters.
        height: Floor-to-floor height in meters.
        floors: Number of floors.
    """
    return f"""Create a {floors}-storey building envelope as a CellComplex using TopologicPy:

1. Create a rectangular prism for each floor:
   - Width: {width}m, Length: {length}m, Height: {height}m
   - Stack them vertically: floor_0 at z=0, floor_1 at z={height}, etc.
2. Merge all floor cells into a CellComplex named "building"
3. Create the dual graph showing floor adjacency
4. Query the building to show the final topology details
5. Set dictionary metadata with building_name="Building", use="office"

The floors should share faces at their interfaces (use CellComplex.ByCells).
Name each floor cell "floor_0", "floor_1", etc."""


@mcp.prompt()
def space_adjacency_analysis(
    topology_name: str = "building",
) -> str:
    """Generate a prompt to analyze spatial adjacency in a topology.

    Args:
        topology_name: Name of the topology to analyze.
    """
    return f"""Analyze the spatial adjacency of '{topology_name}':

1. Extract all cells and store them with prefix "space"
2. Create the adjacency graph with direct=True and via_shared_faces=True
3. For each cell, query its properties (volume, centroid)
4. List all adjacency relationships (which spaces share faces)
5. Identify any isolated spaces (not connected to others)

This is useful for understanding spatial connectivity in building models."""


@mcp.prompt()
def parametric_grid(
    rows: int = 3,
    cols: int = 3,
    cell_width: float = 5.0,
    cell_length: float = 5.0,
    cell_height: float = 3.0,
) -> str:
    """Generate a prompt to create a parametric grid of cells.

    Args:
        rows: Number of rows.
        cols: Number of columns.
        cell_width: Width of each cell.
        cell_length: Length of each cell.
        cell_height: Height of each cell.
    """
    return f"""Create a {rows}x{cols} parametric grid of cells:

For each row i (0 to {rows - 1}) and column j (0 to {cols - 1}):
1. Create an origin vertex at ({cell_width}*j, {cell_length}*i, 0)
2. Create a prism cell with width={cell_width}, length={cell_length}, height={cell_height}
   at that origin, placement="bottom"
3. Name it "cell_{{i}}_{{j}}"

Then:
4. Merge all cells into a CellComplex named "grid"
5. Create the adjacency graph named "grid_graph"
6. Query the grid to show total topology counts"""


# ═══════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    mcp.run()
