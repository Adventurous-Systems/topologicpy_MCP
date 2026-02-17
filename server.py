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
    description=(
        "MCP server for TopologicPy — a spatial modeling library for "
        "creating hierarchical, topological representations of architectural "
        "spaces, buildings, and artefacts using non-manifold topology (NMT). "
        "Supports Vertex, Edge, Wire, Face, Shell, Cell, CellComplex, Cluster, "
        "Graph, Dictionary, and Boolean operations, plus IFC/OBJ/BREP import/export."
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
    """Create a Vertex (point) at the given coordinates and store it with the given name.

    Args:
        name: Unique name to reference this vertex later.
        x: X coordinate. Default 0.
        y: Y coordinate. Default 0.
        z: Z coordinate. Default 0.

    Returns:
        Confirmation with vertex details.
    """
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
    """Create an Edge (line segment) between two named vertices.

    Args:
        name: Unique name for this edge.
        start_vertex: Name of the start vertex.
        end_vertex: Name of the end vertex.
        tolerance: Geometric tolerance. Default 0.0001.

    Returns:
        Confirmation with edge details.
    """
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
    """Create a Wire (polyline) from an ordered list of named vertices.

    Args:
        name: Unique name for this wire.
        vertex_names: Ordered list of vertex names to connect.
        close: If True, close the wire (connect last to first). Default True.
        tolerance: Geometric tolerance. Default 0.0001.

    Returns:
        Confirmation with wire details.
    """
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
    """Create a Face from a closed Wire (external boundary).

    Args:
        name: Unique name for this face.
        wire_name: Name of the closed wire to use as boundary.
        tolerance: Geometric tolerance. Default 0.0001.

    Returns:
        Confirmation with face details.
    """
    from topologicpy.Face import Face
    store: TopologyStore = ctx.request_context.lifespan_context
    wire = store.get(wire_name)
    f = Face.ByWire(wire, tolerance=tolerance)
    store.put(name, f, {"boundary_wire": wire_name})
    return f"Created Face '{name}' from Wire '{wire_name}'"


@mcp.tool()
def create_face_rectangle(
    ctx: Context,
    name: str,
    origin_name: str | None = None,
    width: float = 1.0,
    length: float = 1.0,
    direction: list[float] | None = None,
    tolerance: float = 0.0001,
) -> str:
    """Create a rectangular Face.

    Args:
        name: Unique name for this face.
        origin_name: Name of origin vertex. Default creates at (0,0,0).
        width: Width of rectangle. Default 1.0.
        length: Length of rectangle. Default 1.0.
        direction: Normal direction as [x,y,z]. Default [0,0,1].
        tolerance: Geometric tolerance. Default 0.0001.

    Returns:
        Confirmation with face details.
    """
    from topologicpy.Face import Face
    from topologicpy.Vertex import Vertex
    store: TopologyStore = ctx.request_context.lifespan_context
    origin = store.get(origin_name) if origin_name else Vertex.ByCoordinates(0, 0, 0)
    direction = direction or [0, 0, 1]
    f = Face.Rectangle(
        origin=origin, width=width, length=length,
        direction=direction, tolerance=tolerance,
    )
    store.put(name, f, {"width": width, "length": length, "direction": direction})
    return f"Created rectangular Face '{name}' ({width} x {length})"


@mcp.tool()
def create_face_circle(
    ctx: Context,
    name: str,
    origin_name: str | None = None,
    radius: float = 0.5,
    sides: int = 32,
    direction: list[float] | None = None,
    tolerance: float = 0.0001,
) -> str:
    """Create a circular Face (approximated polygon).

    Args:
        name: Unique name for this face.
        origin_name: Name of origin vertex. Default creates at (0,0,0).
        radius: Radius of the circle. Default 0.5.
        sides: Number of polygon sides. Default 32.
        direction: Normal direction as [x,y,z]. Default [0,0,1].
        tolerance: Geometric tolerance. Default 0.0001.

    Returns:
        Confirmation with face details.
    """
    from topologicpy.Face import Face
    from topologicpy.Vertex import Vertex
    store: TopologyStore = ctx.request_context.lifespan_context
    origin = store.get(origin_name) if origin_name else Vertex.ByCoordinates(0, 0, 0)
    direction = direction or [0, 0, 1]
    f = Face.Circle(
        origin=origin, radius=radius, sides=sides,
        direction=direction, tolerance=tolerance,
    )
    store.put(name, f, {"radius": radius, "sides": sides})
    return f"Created circular Face '{name}' (radius={radius}, sides={sides})"


@mcp.tool()
def create_cell_by_faces(
    ctx: Context,
    name: str,
    face_names: list[str],
    tolerance: float = 0.0001,
) -> str:
    """Create a Cell (3D solid volume) from a list of named Faces.

    The faces must form a closed, watertight shell.

    Args:
        name: Unique name for this cell.
        face_names: List of face names forming the cell boundary.
        tolerance: Geometric tolerance. Default 0.0001.

    Returns:
        Confirmation with cell details.
    """
    from topologicpy.Cell import Cell
    store: TopologyStore = ctx.request_context.lifespan_context
    faces = [store.get(fn) for fn in face_names]
    c = Cell.ByFaces(faces, tolerance=tolerance)
    store.put(name, c, {"faces": face_names})
    return f"Created Cell '{name}' from {len(face_names)} faces"


@mcp.tool()
def create_cell_prism(
    ctx: Context,
    name: str,
    origin_name: str | None = None,
    width: float = 1.0,
    length: float = 1.0,
    height: float = 1.0,
    placement: str = "center",
    tolerance: float = 0.0001,
) -> str:
    """Create a rectangular prism (box) Cell.

    Args:
        name: Unique name for this cell.
        origin_name: Name of origin vertex. Default at (0,0,0).
        width: Width (X). Default 1.0.
        length: Length (Y). Default 1.0.
        height: Height (Z). Default 1.0.
        placement: "center", "bottom", or "lowerleft". Default "center".
        tolerance: Geometric tolerance. Default 0.0001.

    Returns:
        Confirmation with cell details.
    """
    from topologicpy.Cell import Cell
    from topologicpy.Vertex import Vertex
    store: TopologyStore = ctx.request_context.lifespan_context
    origin = store.get(origin_name) if origin_name else Vertex.ByCoordinates(0, 0, 0)
    c = Cell.Prism(
        origin=origin, width=width, length=length, height=height,
        placement=placement, tolerance=tolerance,
    )
    store.put(name, c, {"width": width, "length": length, "height": height})
    return f"Created prism Cell '{name}' ({width} x {length} x {height})"


@mcp.tool()
def create_cell_cylinder(
    ctx: Context,
    name: str,
    origin_name: str | None = None,
    radius: float = 0.5,
    height: float = 1.0,
    sides: int = 32,
    placement: str = "center",
    tolerance: float = 0.0001,
) -> str:
    """Create a cylindrical Cell.

    Args:
        name: Unique name for this cell.
        origin_name: Name of origin vertex. Default at (0,0,0).
        radius: Radius of the cylinder. Default 0.5.
        height: Height of the cylinder. Default 1.0.
        sides: Number of polygon sides. Default 32.
        placement: "center", "bottom", or "lowerleft". Default "center".
        tolerance: Geometric tolerance. Default 0.0001.

    Returns:
        Confirmation with cell details.
    """
    from topologicpy.Cell import Cell
    from topologicpy.Vertex import Vertex
    store: TopologyStore = ctx.request_context.lifespan_context
    origin = store.get(origin_name) if origin_name else Vertex.ByCoordinates(0, 0, 0)
    c = Cell.Cylinder(
        origin=origin, radius=radius, height=height, sides=sides,
        placement=placement, tolerance=tolerance,
    )
    store.put(name, c, {"radius": radius, "height": height, "sides": sides})
    return f"Created cylindrical Cell '{name}' (r={radius}, h={height})"


@mcp.tool()
def create_cellcomplex_by_cells(
    ctx: Context,
    name: str,
    cell_names: list[str],
    tolerance: float = 0.0001,
) -> str:
    """Create a CellComplex from a list of named Cells.

    Cells are merged, sharing faces/edges/vertices at intersections.

    Args:
        name: Unique name for this cell complex.
        cell_names: List of cell names to merge.
        tolerance: Geometric tolerance. Default 0.0001.

    Returns:
        Confirmation with cell complex details.
    """
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
    """Create a Cluster (unstructured collection) from named topologies.

    Args:
        name: Unique name for this cluster.
        topology_names: List of topology names to include.

    Returns:
        Confirmation with cluster details.
    """
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
def boolean_union(
    ctx: Context,
    name: str,
    topology_a: str,
    topology_b: str,
    tolerance: float = 0.0001,
) -> str:
    """Compute the Boolean union of two topologies.

    Args:
        name: Name for the resulting topology.
        topology_a: Name of the first topology.
        topology_b: Name of the second topology.
        tolerance: Geometric tolerance. Default 0.0001.

    Returns:
        Confirmation of the union result.
    """
    from topologicpy.Topology import Topology
    store: TopologyStore = ctx.request_context.lifespan_context
    a = store.get(topology_a)
    b = store.get(topology_b)
    result = Topology.Union(a, b, tolerance=tolerance)
    store.put(name, result, {"operation": "union", "a": topology_a, "b": topology_b})
    return f"Created '{name}' = Union('{topology_a}', '{topology_b}') → {Topology.TypeAsString(result)}"


@mcp.tool()
def boolean_difference(
    ctx: Context,
    name: str,
    topology_a: str,
    topology_b: str,
    tolerance: float = 0.0001,
) -> str:
    """Compute the Boolean difference (A minus B) of two topologies.

    Args:
        name: Name for the resulting topology.
        topology_a: Name of the topology to subtract from.
        topology_b: Name of the topology to subtract.
        tolerance: Geometric tolerance. Default 0.0001.

    Returns:
        Confirmation of the difference result.
    """
    from topologicpy.Topology import Topology
    store: TopologyStore = ctx.request_context.lifespan_context
    a = store.get(topology_a)
    b = store.get(topology_b)
    result = Topology.Difference(a, b, tolerance=tolerance)
    store.put(name, result, {"operation": "difference", "a": topology_a, "b": topology_b})
    return f"Created '{name}' = Difference('{topology_a}', '{topology_b}') → {Topology.TypeAsString(result)}"


@mcp.tool()
def boolean_intersect(
    ctx: Context,
    name: str,
    topology_a: str,
    topology_b: str,
    tolerance: float = 0.0001,
) -> str:
    """Compute the Boolean intersection of two topologies.

    Args:
        name: Name for the resulting topology.
        topology_a: Name of the first topology.
        topology_b: Name of the second topology.
        tolerance: Geometric tolerance. Default 0.0001.

    Returns:
        Confirmation of the intersection result.
    """
    from topologicpy.Topology import Topology
    store: TopologyStore = ctx.request_context.lifespan_context
    a = store.get(topology_a)
    b = store.get(topology_b)
    result = Topology.Intersect(a, b, tolerance=tolerance)
    store.put(name, result, {"operation": "intersect", "a": topology_a, "b": topology_b})
    return f"Created '{name}' = Intersect('{topology_a}', '{topology_b}') → {Topology.TypeAsString(result)}"


@mcp.tool()
def self_merge(
    ctx: Context,
    name: str,
    topology_name: str,
    tolerance: float = 0.0001,
) -> str:
    """Merge a topology with itself to resolve self-intersections and shared boundaries.

    This is essential for creating valid CellComplexes from overlapping cells.

    Args:
        name: Name for the merged result.
        topology_name: Name of the topology to self-merge.
        tolerance: Geometric tolerance. Default 0.0001.

    Returns:
        Confirmation of the merge result.
    """
    from topologicpy.Topology import Topology
    store: TopologyStore = ctx.request_context.lifespan_context
    topo = store.get(topology_name)
    result = Topology.SelfMerge(topo, tolerance=tolerance)
    store.put(name, result, {"operation": "self_merge", "source": topology_name})
    return f"Created '{name}' = SelfMerge('{topology_name}') → {Topology.TypeAsString(result)}"


# ═══════════════════════════════════════════════════════════════════════════
# TOOLS — Transformations
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
def translate(
    ctx: Context,
    name: str,
    topology_name: str,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
) -> str:
    """Translate (move) a topology by the given vector.

    Args:
        name: Name for the translated copy.
        topology_name: Name of the topology to translate.
        x: Translation in X. Default 0.
        y: Translation in Y. Default 0.
        z: Translation in Z. Default 0.

    Returns:
        Confirmation of the translation.
    """
    from topologicpy.Topology import Topology
    store: TopologyStore = ctx.request_context.lifespan_context
    topo = store.get(topology_name)
    result = Topology.Translate(topo, x, y, z)
    store.put(name, result, {"operation": "translate", "source": topology_name, "vector": [x, y, z]})
    return f"Created '{name}' = Translate('{topology_name}', [{x}, {y}, {z}])"


@mcp.tool()
def rotate(
    ctx: Context,
    name: str,
    topology_name: str,
    origin_name: str | None = None,
    axis: list[float] | None = None,
    angle: float = 0.0,
) -> str:
    """Rotate a topology around an axis through an origin point.

    Args:
        name: Name for the rotated copy.
        topology_name: Name of the topology to rotate.
        origin_name: Name of the rotation center vertex. Default (0,0,0).
        axis: Rotation axis as [x,y,z]. Default [0,0,1].
        angle: Rotation angle in degrees. Default 0.

    Returns:
        Confirmation of the rotation.
    """
    from topologicpy.Topology import Topology
    from topologicpy.Vertex import Vertex
    store: TopologyStore = ctx.request_context.lifespan_context
    topo = store.get(topology_name)
    origin = store.get(origin_name) if origin_name else Vertex.ByCoordinates(0, 0, 0)
    axis = axis or [0, 0, 1]
    result = Topology.Rotate(topo, origin=origin, axis=axis, angle=angle)
    store.put(name, result, {"operation": "rotate", "source": topology_name, "angle": angle})
    return f"Created '{name}' = Rotate('{topology_name}', angle={angle}°, axis={axis})"


@mcp.tool()
def scale(
    ctx: Context,
    name: str,
    topology_name: str,
    origin_name: str | None = None,
    x_factor: float = 1.0,
    y_factor: float = 1.0,
    z_factor: float = 1.0,
) -> str:
    """Scale a topology relative to an origin point.

    Args:
        name: Name for the scaled copy.
        topology_name: Name of the topology to scale.
        origin_name: Name of the scale center vertex. Default (0,0,0).
        x_factor: Scale factor in X. Default 1.0.
        y_factor: Scale factor in Y. Default 1.0.
        z_factor: Scale factor in Z. Default 1.0.

    Returns:
        Confirmation of the scaling.
    """
    from topologicpy.Topology import Topology
    from topologicpy.Vertex import Vertex
    store: TopologyStore = ctx.request_context.lifespan_context
    topo = store.get(topology_name)
    origin = store.get(origin_name) if origin_name else Vertex.ByCoordinates(0, 0, 0)
    result = Topology.Scale(topo, origin=origin, x=x_factor, y=y_factor, z=z_factor)
    store.put(name, result, {"operation": "scale", "source": topology_name})
    return f"Created '{name}' = Scale('{topology_name}', factors=[{x_factor}, {y_factor}, {z_factor}])"


# ═══════════════════════════════════════════════════════════════════════════
# TOOLS — Query & Analysis
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
def list_topologies(ctx: Context) -> str:
    """List all named topologies currently in the session store.

    Returns:
        JSON array of objects with name, type, and metadata.
    """
    store: TopologyStore = ctx.request_context.lifespan_context
    items = store.list_all()
    if not items:
        return "Session store is empty. Create some topologies first."
    return json.dumps(items, indent=2)


@mcp.tool()
def query_topology(
    ctx: Context,
    topology_name: str,
) -> str:
    """Get detailed information about a named topology.

    Returns type, sub-topology counts, centroid, bounding box, and volume/area.

    Args:
        topology_name: Name of the topology to query.

    Returns:
        JSON object with topology details.
    """
    from topologicpy.Topology import Topology
    from topologicpy.Vertex import Vertex
    from topologicpy.Cell import Cell
    from topologicpy.Face import Face
    store: TopologyStore = ctx.request_context.lifespan_context
    topo = store.get(topology_name)
    topo_type = Topology.TypeAsString(topo)

    info: dict[str, Any] = {
        "name": topology_name,
        "type": topo_type,
    }

    # Sub-topology counts
    vertices = Topology.Vertices(topo) or []
    edges = Topology.Edges(topo) or []
    faces = Topology.Faces(topo) or []
    cells = Topology.Cells(topo) or []
    info["counts"] = {
        "vertices": len(vertices),
        "edges": len(edges),
        "faces": len(faces),
        "cells": len(cells),
    }

    # Centroid
    centroid = Topology.Centroid(topo)
    if centroid:
        info["centroid"] = {
            "x": round(Vertex.X(centroid), 6),
            "y": round(Vertex.Y(centroid), 6),
            "z": round(Vertex.Z(centroid), 6),
        }

    # Bounding box
    try:
        bb = Topology.BoundingBox(topo)
        if bb:
            info["bounding_box"] = str(bb)
    except Exception:
        pass

    # Volume (for cells)
    if topo_type == "Cell":
        try:
            vol = Cell.Volume(topo)
            info["volume"] = round(vol, 6)
        except Exception:
            pass

    # Area (for faces)
    if topo_type == "Face":
        try:
            area = Face.Area(topo)
            info["area"] = round(area, 6)
        except Exception:
            pass

    return json.dumps(info, indent=2)


@mcp.tool()
def get_vertices(
    ctx: Context,
    topology_name: str,
    mantissa: int = 6,
) -> str:
    """Get all vertex coordinates of a topology.

    Args:
        topology_name: Name of the topology.
        mantissa: Decimal places for coordinates. Default 6.

    Returns:
        JSON array of [x, y, z] coordinates.
    """
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
    """Extract sub-topologies (vertices, edges, faces, cells) from a topology.

    Optionally stores each sub-topology with a numbered name prefix.

    Args:
        topology_name: Name of the parent topology.
        sub_type: Type to extract — "vertex", "edge", "wire", "face", "shell", "cell".
        store_with_prefix: If provided, store each sub-topology as "{prefix}_0", "{prefix}_1", etc.

    Returns:
        Count of extracted sub-topologies and their names if stored.
    """
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
    return json.dumps(result, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# TOOLS — Graph Operations
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
def create_graph_from_topology(
    ctx: Context,
    name: str,
    topology_name: str,
    direct: bool = True,
    via_shared_faces: bool = False,
    via_shared_edges: bool = False,
    via_shared_vertices: bool = False,
    to_exterior_faces: bool = False,
    to_exterior_edges: bool = False,
    to_exterior_vertices: bool = False,
    tolerance: float = 0.0001,
) -> str:
    """Create a dual Graph from a topology (e.g. adjacency graph of cells in a CellComplex).

    Args:
        name: Name for the resulting graph.
        topology_name: Name of the source topology.
        direct: Include direct adjacency. Default True.
        via_shared_faces: Connect through shared faces. Default False.
        via_shared_edges: Connect through shared edges. Default False.
        via_shared_vertices: Connect through shared vertices. Default False.
        to_exterior_faces: Include exterior face connections. Default False.
        to_exterior_edges: Include exterior edge connections. Default False.
        to_exterior_vertices: Include exterior vertex connections. Default False.
        tolerance: Geometric tolerance. Default 0.0001.

    Returns:
        Confirmation with graph details.
    """
    from topologicpy.Graph import Graph
    store: TopologyStore = ctx.request_context.lifespan_context
    topo = store.get(topology_name)
    g = Graph.ByTopology(
        topo,
        direct=direct,
        viaSharedTopologies=via_shared_faces,
        viaSharedEdges=via_shared_edges,
        viaSharedVertices=via_shared_vertices,
        toExteriorTopologies=to_exterior_faces,
        toExteriorEdges=to_exterior_edges,
        toExteriorVertices=to_exterior_vertices,
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
    """Find the shortest path between two vertices in a graph.

    Args:
        name: Name to store the resulting path topology.
        graph_name: Name of the graph.
        start_vertex_name: Name of the start vertex.
        end_vertex_name: Name of the end vertex.
        tolerance: Geometric tolerance. Default 0.0001.

    Returns:
        Path details (length, vertex count).
    """
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
def set_dictionary(
    ctx: Context,
    topology_name: str,
    data: dict[str, Any],
) -> str:
    """Attach a dictionary (key-value metadata) to a topology.

    Args:
        topology_name: Name of the topology.
        data: Dictionary of key-value pairs to attach.

    Returns:
        Confirmation.
    """
    from topologicpy.Topology import Topology
    from topologicpy.Dictionary import Dictionary
    store: TopologyStore = ctx.request_context.lifespan_context
    topo = store.get(topology_name)
    keys = list(data.keys())
    values = list(data.values())
    d = Dictionary.ByKeysValues(keys, values)
    topo = Topology.SetDictionary(topo, d)
    store.objects[topology_name] = topo  # update in-place
    return f"Set dictionary on '{topology_name}' with keys: {keys}"


@mcp.tool()
def get_dictionary(
    ctx: Context,
    topology_name: str,
) -> str:
    """Get the dictionary (metadata) attached to a topology.

    Args:
        topology_name: Name of the topology.

    Returns:
        JSON object of the dictionary contents.
    """
    from topologicpy.Topology import Topology
    from topologicpy.Dictionary import Dictionary
    store: TopologyStore = ctx.request_context.lifespan_context
    topo = store.get(topology_name)
    d = Topology.Dictionary(topo)
    if d is None:
        return f"No dictionary attached to '{topology_name}'"
    keys = Dictionary.Keys(d) or []
    values = Dictionary.Values(d) or []
    return json.dumps(dict(zip(keys, values)), indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════════
# TOOLS — Import / Export
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
def export_brep(
    ctx: Context,
    topology_name: str,
    file_path: str | None = None,
) -> str:
    """Export a topology as a BREP string (OpenCASCADE format).

    Args:
        topology_name: Name of the topology to export.
        file_path: Optional file path to save the BREP. If None, returns the string.

    Returns:
        The BREP string or confirmation of file save.
    """
    from topologicpy.Topology import Topology
    store: TopologyStore = ctx.request_context.lifespan_context
    topo = store.get(topology_name)
    brep_string = Topology.BREPString(topo)
    if file_path:
        with open(file_path, "w") as f:
            f.write(brep_string)
        return f"Exported '{topology_name}' BREP to {file_path} ({len(brep_string)} chars)"
    return brep_string


@mcp.tool()
def import_brep(
    ctx: Context,
    name: str,
    brep_string: str | None = None,
    file_path: str | None = None,
) -> str:
    """Import a topology from a BREP string or file.

    Args:
        name: Name to assign to the imported topology.
        brep_string: BREP string content. Mutually exclusive with file_path.
        file_path: Path to a .brep file. Mutually exclusive with brep_string.

    Returns:
        Confirmation with imported topology type.
    """
    from topologicpy.Topology import Topology
    store: TopologyStore = ctx.request_context.lifespan_context
    if file_path:
        with open(file_path, "r") as f:
            brep_string = f.read()
    if not brep_string:
        return "Error: provide either brep_string or file_path"
    topo = Topology.ByBREPString(brep_string)
    store.put(name, topo, {"source": "brep_import"})
    return f"Imported '{name}' as {Topology.TypeAsString(topo)}"


@mcp.tool()
def export_obj(
    ctx: Context,
    topology_name: str,
    file_path: str,
    transpose: bool = False,
    tolerance: float = 0.0001,
) -> str:
    """Export a topology as an OBJ file (triangulated mesh).

    Args:
        topology_name: Name of the topology to export.
        file_path: File path for the OBJ output.
        transpose: If True, swap Y and Z axes. Default False.
        tolerance: Geometric tolerance. Default 0.0001.

    Returns:
        Confirmation of export.
    """
    from topologicpy.Topology import Topology
    store: TopologyStore = ctx.request_context.lifespan_context
    topo = store.get(topology_name)
    Topology.ExportToOBJ(topo, path=file_path, transposeAxes=transpose, tolerance=tolerance)
    return f"Exported '{topology_name}' to OBJ: {file_path}"


@mcp.tool()
def export_ifc(
    ctx: Context,
    topology_name: str,
    file_path: str,
    building_name: str = "Default Building",
) -> str:
    """Export a topology as an IFC file.

    Args:
        topology_name: Name of the topology to export.
        file_path: File path for the IFC output.
        building_name: Name for the IFC building entity. Default "Default Building".

    Returns:
        Confirmation of export.
    """
    from topologicpy.Topology import Topology
    store: TopologyStore = ctx.request_context.lifespan_context
    topo = store.get(topology_name)
    Topology.ExportToIFC(topo, path=file_path, buildingName=building_name)
    return f"Exported '{topology_name}' to IFC: {file_path}"


@mcp.tool()
def import_ifc(
    ctx: Context,
    name: str,
    file_path: str,
    tolerance: float = 0.0001,
) -> str:
    """Import a topology from an IFC file.

    Args:
        name: Name to assign to the imported topology.
        file_path: Path to the IFC file.
        tolerance: Geometric tolerance. Default 0.0001.

    Returns:
        Confirmation with imported topology details.
    """
    from topologicpy.Topology import Topology
    store: TopologyStore = ctx.request_context.lifespan_context
    topo = Topology.ByIFCFile(file_path, tolerance=tolerance)
    store.put(name, topo, {"source": "ifc_import", "file": file_path})
    return f"Imported '{name}' from IFC as {Topology.TypeAsString(topo)}"


# ═══════════════════════════════════════════════════════════════════════════
# TOOLS — Session Management
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
def remove_topology(
    ctx: Context,
    topology_name: str,
) -> str:
    """Remove a named topology from the session store.

    Args:
        topology_name: Name of the topology to remove.

    Returns:
        Confirmation of removal.
    """
    store: TopologyStore = ctx.request_context.lifespan_context
    if store.remove(topology_name):
        return f"Removed '{topology_name}' from store."
    return f"'{topology_name}' not found in store."


@mcp.tool()
def rename_topology(
    ctx: Context,
    old_name: str,
    new_name: str,
) -> str:
    """Rename a topology in the session store.

    Args:
        old_name: Current name.
        new_name: New name.

    Returns:
        Confirmation of rename.
    """
    store: TopologyStore = ctx.request_context.lifespan_context
    topo = store.get(old_name)
    meta = store.metadata.get(old_name, {})
    store.remove(old_name)
    store.put(new_name, topo, meta)
    return f"Renamed '{old_name}' → '{new_name}'"


@mcp.tool()
def copy_topology(
    ctx: Context,
    source_name: str,
    new_name: str,
) -> str:
    """Create a deep copy of a topology under a new name.

    Args:
        source_name: Name of the topology to copy.
        new_name: Name for the copy.

    Returns:
        Confirmation of copy.
    """
    from topologicpy.Topology import Topology
    store: TopologyStore = ctx.request_context.lifespan_context
    topo = store.get(source_name)
    copy = Topology.Copy(topo)
    meta = store.metadata.get(source_name, {}).copy()
    meta["copied_from"] = source_name
    store.put(new_name, copy, meta)
    return f"Copied '{source_name}' → '{new_name}'"


# ═══════════════════════════════════════════════════════════════════════════
# RESOURCES — Model State
# ═══════════════════════════════════════════════════════════════════════════

@mcp.resource("topologic://session/summary")
def session_summary(ctx: Context) -> str:
    """Summary of all topologies in the current session."""
    store: TopologyStore = ctx.request_context.lifespan_context
    items = store.list_all()
    if not items:
        return "Session is empty."
    return json.dumps(items, indent=2)


@mcp.resource("topologic://topology/{name}/brep")
def topology_brep_resource(name: str, ctx: Context) -> str:
    """Get the BREP string for a named topology."""
    from topologicpy.Topology import Topology
    store: TopologyStore = ctx.request_context.lifespan_context
    topo = store.get(name)
    return Topology.BREPString(topo)


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
