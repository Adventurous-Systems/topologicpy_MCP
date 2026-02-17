"""
Integration test: 3-floor building, 9 rooms/floor, windows + doors.
Exercises every merged tool to validate the P1 refactor.

Building layout (3x3 grid per floor):
  +---+---+---+
  | 0 | 1 | 2 |   Each room: 5m x 5m x 3m
  +---+---+---+   3 floors stacked vertically
  | 3 | 4 | 5 |   Windows: small face on exterior wall of each room
  +---+---+---+   Doors: openings between adjacent rooms
  | 6 | 7 | 8 |
  +---+---+---+
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass

# Mock the MCP Context so we can call tool functions directly
from server import (
    TopologyStore,
    create_vertex, create_edge, create_wire, create_face_by_wire,
    create_face_shape, create_cell_shape, create_cell_by_faces,
    create_cellcomplex_by_cells, create_cluster,
    boolean, transform, query_topology, list_topologies,
    get_vertices, get_sub_topologies,
    dictionary, manage_topology, export_topology,
    create_graph_from_topology, graph_shortest_path,
)

store = TopologyStore()

@dataclass
class MockRequestContext:
    lifespan_context: TopologyStore

@dataclass
class MockContext:
    request_context: MockRequestContext

ctx = MockContext(request_context=MockRequestContext(lifespan_context=store))

# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────
ROOM_W, ROOM_L, ROOM_H = 5.0, 5.0, 3.0
ROWS, COLS, FLOORS = 3, 3, 3
WINDOW_W, WINDOW_H = 1.2, 1.0
DOOR_W, DOOR_H = 0.9, 2.1

passed = 0
failed = 0

def check(label, result, must_contain=None):
    global passed, failed
    ok = True
    if must_contain and must_contain not in result:
        ok = False
    if ok:
        passed += 1
        print(f"  ✓ {label}")
    else:
        failed += 1
        print(f"  ✗ {label}  →  got: {result}")
    return result


# ══════════════════════════════════════════════════════════════════
# STEP 1 — Create rooms (27 cells = 3 floors × 9 rooms)
# ══════════════════════════════════════════════════════════════════
print("\n═══ STEP 1: Create rooms (create_cell_shape, create_vertex, transform) ═══")

room_names = []
for floor in range(FLOORS):
    for row in range(ROWS):
        for col in range(COLS):
            rname = f"room_f{floor}_r{row}_c{col}"
            room_names.append(rname)

            # Create origin vertex for this room
            ox = col * ROOM_W
            oy = row * ROOM_L
            oz = floor * ROOM_H
            oname = f"origin_{rname}"
            check(
                f"vertex {oname}",
                create_vertex(ctx, name=oname, x=ox, y=oy, z=oz),
                "Created Vertex",
            )

            # Create prism cell at that origin
            check(
                f"cell {rname}",
                create_cell_shape(
                    ctx, name=rname, shape="prism",
                    origin_name=oname,
                    width=ROOM_W, length=ROOM_L, height=ROOM_H,
                    placement="bottom",
                ),
                "Created prism Cell",
            )

print(f"\n  → {len(room_names)} rooms created")

# ══════════════════════════════════════════════════════════════════
# STEP 2 — Merge each floor into a CellComplex
# ══════════════════════════════════════════════════════════════════
print("\n═══ STEP 2: Merge floors (create_cellcomplex_by_cells) ═══")

floor_names = []
for floor in range(FLOORS):
    fname = f"floor_{floor}"
    floor_rooms = [r for r in room_names if f"_f{floor}_" in r]
    check(
        f"cellcomplex {fname}",
        create_cellcomplex_by_cells(ctx, name=fname, cell_names=floor_rooms),
        "Created CellComplex",
    )
    floor_names.append(fname)

# ══════════════════════════════════════════════════════════════════
# STEP 3 — Stack floors into building via boolean self_merge
# ══════════════════════════════════════════════════════════════════
print("\n═══ STEP 3: Assemble building (create_cluster, boolean self_merge) ═══")

check(
    "cluster all floors",
    create_cluster(ctx, name="all_floors", topology_names=floor_names),
    "Created Cluster",
)

check(
    "self_merge → building",
    boolean(ctx, name="building", operation="self_merge", topology_a="all_floors"),
    "SelfMerge",
)

# ══════════════════════════════════════════════════════════════════
# STEP 4 — Query the building (query_topology — summary + full)
# ══════════════════════════════════════════════════════════════════
print("\n═══ STEP 4: Query building (query_topology summary + full) ═══")

check(
    "query summary",
    query_topology(ctx, topology_name="building", detail="summary"),
    "building:",
)

full = check(
    "query full",
    query_topology(ctx, topology_name="building", detail="full"),
    '"type"',
)
print(f"    full query result: {full[:200]}...")

# ══════════════════════════════════════════════════════════════════
# STEP 5 — Create windows (create_face_shape rectangle on each room)
# ══════════════════════════════════════════════════════════════════
print("\n═══ STEP 5: Create windows (create_face_shape rectangle, transform) ═══")

window_names = []
for floor in range(FLOORS):
    for row in range(ROWS):
        for col in range(COLS):
            wname = f"window_f{floor}_r{row}_c{col}"
            window_names.append(wname)

            # Create a window as a small rectangle in the XZ plane (normal along Y)
            check(
                f"face {wname}",
                create_face_shape(
                    ctx, name=wname, shape="rectangle",
                    width=WINDOW_W, length=WINDOW_H,
                    direction=[0, 1, 0],  # normal along Y (exterior wall)
                ),
                "Created rectangular Face",
            )

            # Position the window on the exterior wall of this room
            wx = col * ROOM_W + ROOM_W / 2
            wy = row * ROOM_L  # front wall
            wz = floor * ROOM_H + 1.5  # ~middle height
            tname = f"{wname}_placed"
            check(
                f"translate {wname}",
                transform(
                    ctx, name=tname, topology_name=wname,
                    operation="translate", x=wx, y=wy, z=wz,
                ),
                "Translate",
            )
            window_names[-1] = tname  # track placed version

print(f"\n  → {len(window_names)} windows created and placed")

# ══════════════════════════════════════════════════════════════════
# STEP 6 — Create doors between adjacent rooms (same floor)
# ══════════════════════════════════════════════════════════════════
print("\n═══ STEP 6: Create doors (create_face_shape, transform) ═══")

door_names = []
door_id = 0
for floor in range(FLOORS):
    for row in range(ROWS):
        for col in range(COLS):
            # Door to the right neighbor (col+1)
            if col < COLS - 1:
                dname = f"door_{door_id}"
                door_id += 1
                # Door in the YZ plane (normal along X), between col and col+1
                check(
                    f"face {dname}",
                    create_face_shape(
                        ctx, name=dname, shape="rectangle",
                        width=DOOR_W, length=DOOR_H,
                        direction=[1, 0, 0],
                    ),
                    "Created rectangular Face",
                )
                dx = (col + 1) * ROOM_W  # on the shared wall
                dy = row * ROOM_L + ROOM_L / 2
                dz = floor * ROOM_H + DOOR_H / 2
                pname = f"{dname}_placed"
                check(
                    f"translate {dname}",
                    transform(ctx, name=pname, topology_name=dname,
                              operation="translate", x=dx, y=dy, z=dz),
                    "Translate",
                )
                door_names.append(pname)

            # Door to the neighbor behind (row+1)
            if row < ROWS - 1:
                dname = f"door_{door_id}"
                door_id += 1
                # Door in the XZ plane (normal along Y)
                check(
                    f"face {dname}",
                    create_face_shape(
                        ctx, name=dname, shape="rectangle",
                        width=DOOR_W, length=DOOR_H,
                        direction=[0, 1, 0],
                    ),
                    "Created rectangular Face",
                )
                dx = col * ROOM_W + ROOM_W / 2
                dy = (row + 1) * ROOM_L
                dz = floor * ROOM_H + DOOR_H / 2
                pname = f"{dname}_placed"
                check(
                    f"translate {dname}",
                    transform(ctx, name=pname, topology_name=dname,
                              operation="translate", x=dx, y=dy, z=dz),
                    "Translate",
                )
                door_names.append(pname)

print(f"\n  → {len(door_names)} doors created and placed")

# ══════════════════════════════════════════════════════════════════
# STEP 7 — Group doors + windows into clusters
# ══════════════════════════════════════════════════════════════════
print("\n═══ STEP 7: Cluster doors & windows (create_cluster) ═══")

check(
    "cluster all_windows",
    create_cluster(ctx, name="all_windows", topology_names=window_names),
    "Created Cluster",
)
check(
    "cluster all_doors",
    create_cluster(ctx, name="all_doors", topology_names=door_names),
    "Created Cluster",
)

# ══════════════════════════════════════════════════════════════════
# STEP 8 — Graph analysis (create_graph, shortest_path)
# ══════════════════════════════════════════════════════════════════
print("\n═══ STEP 8: Graph analysis (create_graph_from_topology, graph_shortest_path) ═══")

check(
    "adjacency graph",
    create_graph_from_topology(
        ctx, name="building_graph", topology_name="building",
        direct=True, via_shared_topologies=True,
    ),
    "Created Graph",
)

# ══════════════════════════════════════════════════════════════════
# STEP 9 — Dictionary metadata
# ══════════════════════════════════════════════════════════════════
print("\n═══ STEP 9: Dictionary metadata (dictionary set + get) ═══")

check(
    "set dict",
    dictionary(ctx, topology_name="building", action="set", data={
        "name": "Test Building",
        "floors": FLOORS,
        "rooms_per_floor": ROWS * COLS,
        "total_rooms": FLOORS * ROWS * COLS,
    }),
    "Set dictionary",
)

check(
    "get dict",
    dictionary(ctx, topology_name="building", action="get"),
    "Test Building",
)

# ══════════════════════════════════════════════════════════════════
# STEP 10 — Session management (manage_topology copy + rename)
# ══════════════════════════════════════════════════════════════════
print("\n═══ STEP 10: Session management (manage_topology) ═══")

check(
    "copy building",
    manage_topology(ctx, action="copy", topology_name="building", new_name="building_backup"),
    "Copied",
)

check(
    "rename backup",
    manage_topology(ctx, action="rename", topology_name="building_backup", new_name="building_v2"),
    "Renamed",
)

check(
    "remove building_v2",
    manage_topology(ctx, action="remove", topology_name="building_v2"),
    "Removed",
)

# ══════════════════════════════════════════════════════════════════
# STEP 11 — Sub-topology extraction + vertex query
# ══════════════════════════════════════════════════════════════════
print("\n═══ STEP 11: Sub-topology extraction (get_sub_topologies, get_vertices) ═══")

check(
    "extract faces",
    get_sub_topologies(ctx, topology_name="building", sub_type="face"),
    '"count"',
)

check(
    "extract cells",
    get_sub_topologies(ctx, topology_name="building", sub_type="cell", store_with_prefix="bldg_cell"),
    '"stored_as"',
)

check(
    "get vertices of room",
    get_vertices(ctx, topology_name="room_f0_r0_c0"),
    "[",
)

# ══════════════════════════════════════════════════════════════════
# STEP 12 — Export
# ══════════════════════════════════════════════════════════════════
print("\n═══ STEP 12: Export (export_topology brep + obj) ═══")

import tempfile, os

with tempfile.TemporaryDirectory() as tmp:
    brep_path = os.path.join(tmp, "building.brep")
    check(
        "export brep",
        export_topology(ctx, topology_name="building", file_path=brep_path, format="brep"),
        "Exported",
    )
    assert os.path.exists(brep_path), "BREP file not created!"
    print(f"    BREP file size: {os.path.getsize(brep_path):,} bytes")

    obj_path = os.path.join(tmp, "building.obj")
    check(
        "export obj",
        export_topology(ctx, topology_name="building", file_path=obj_path,
                        format="obj", transpose=True),
        "Exported",
    )
    assert os.path.exists(obj_path), "OBJ file not created!"
    print(f"    OBJ file size: {os.path.getsize(obj_path):,} bytes")

# ══════════════════════════════════════════════════════════════════
# STEP 13 — List session
# ══════════════════════════════════════════════════════════════════
print("\n═══ STEP 13: List session (list_topologies) ═══")

listing = list_topologies(ctx)
check("list topologies", listing, "building")
import json
items = json.loads(listing)
print(f"    Total objects in session: {len(items)}")

# ══════════════════════════════════════════════════════════════════
# STEP 14 — Additional merged tools: boolean union/difference/intersect, rotate, scale, circle face, cylinder cell
# ══════════════════════════════════════════════════════════════════
print("\n═══ STEP 14: Remaining merged tool variants ═══")

# Circle face
check(
    "circle face",
    create_face_shape(ctx, name="pillar_profile", shape="circle", radius=0.3, sides=16),
    "Created circular Face",
)

# Cylinder cell
check(
    "cylinder cell",
    create_cell_shape(ctx, name="pillar", shape="cylinder", radius=0.3, height=ROOM_H, sides=16, placement="bottom"),
    "Created cylindrical Cell",
)

# Boolean union
check(
    "boolean union",
    boolean(ctx, name="room_union", operation="union",
            topology_a="room_f0_r0_c0", topology_b="room_f0_r0_c1"),
    "Union",
)

# Boolean difference
check(
    "boolean difference",
    boolean(ctx, name="room_diff", operation="difference",
            topology_a="room_f0_r0_c0", topology_b="pillar"),
    "Difference",
)

# Boolean intersect
check(
    "boolean intersect",
    boolean(ctx, name="room_sect", operation="intersect",
            topology_a="room_f0_r0_c0", topology_b="room_f0_r0_c1"),
    "Intersect",
)

# Rotate
check(
    "rotate pillar",
    transform(ctx, name="pillar_rotated", topology_name="pillar",
              operation="rotate", axis=[1, 0, 0], angle=90),
    "Rotate",
)

# Scale
check(
    "scale pillar",
    transform(ctx, name="pillar_scaled", topology_name="pillar",
              operation="scale", x_factor=2.0, y_factor=2.0, z_factor=1.0),
    "Scale",
)

# ══════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print(f"  RESULTS: {passed} passed, {failed} failed")
print("=" * 64)

if failed > 0:
    sys.exit(1)
else:
    print("  All merged tools working correctly! ✓")
