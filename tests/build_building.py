"""
Build a 3-floor, 9-rooms-per-floor building with windows and doors.
Exports to BREP and OBJ in the project output/ directory.
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass
from server import (
    TopologyStore,
    create_vertex, create_face_shape, create_cell_shape,
    create_cellcomplex_by_cells, create_cluster,
    boolean, transform, query_topology, list_topologies,
    get_sub_topologies, dictionary, export_topology,
    create_graph_from_topology,
)

store = TopologyStore()

@dataclass
class MockRequestContext:
    lifespan_context: TopologyStore

@dataclass
class MockContext:
    request_context: MockRequestContext

ctx = MockContext(request_context=MockRequestContext(lifespan_context=store))

# ── Config ──
ROOM_W, ROOM_L, ROOM_H = 5.0, 5.0, 3.0
ROWS, COLS, FLOORS = 3, 3, 3
WINDOW_W, WINDOW_H = 1.2, 1.0
DOOR_W, DOOR_H = 0.9, 2.1

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# 1. Create 27 rooms (3 floors × 3×3 grid)
# ══════════════════════════════════════════════════════════════
print("Creating 27 rooms...")
room_names = []
for floor in range(FLOORS):
    for row in range(ROWS):
        for col in range(COLS):
            rname = f"room_f{floor}_r{row}_c{col}"
            oname = f"origin_{rname}"
            ox, oy, oz = col * ROOM_W, row * ROOM_L, floor * ROOM_H

            create_vertex(ctx, name=oname, x=ox, y=oy, z=oz)
            create_cell_shape(ctx, name=rname, shape="prism",
                              origin_name=oname,
                              width=ROOM_W, length=ROOM_L, height=ROOM_H,
                              placement="bottom")
            room_names.append(rname)
print(f"  → {len(room_names)} rooms")

# ══════════════════════════════════════════════════════════════
# 2. Merge each floor into a CellComplex
# ══════════════════════════════════════════════════════════════
print("Merging floors...")
floor_names = []
for floor in range(FLOORS):
    fname = f"floor_{floor}"
    floor_rooms = [r for r in room_names if f"_f{floor}_" in r]
    create_cellcomplex_by_cells(ctx, name=fname, cell_names=floor_rooms)
    floor_names.append(fname)
print(f"  → {len(floor_names)} floor CellComplexes")

# ══════════════════════════════════════════════════════════════
# 3. Assemble building
# ══════════════════════════════════════════════════════════════
print("Assembling building...")
create_cluster(ctx, name="all_floors", topology_names=floor_names)
result = boolean(ctx, name="building", operation="self_merge", topology_a="all_floors")
print(f"  → {result}")

# ══════════════════════════════════════════════════════════════
# 4. Create 27 windows (one per room, on front wall)
# ══════════════════════════════════════════════════════════════
print("Creating windows...")
window_names = []
for floor in range(FLOORS):
    for row in range(ROWS):
        for col in range(COLS):
            wname = f"window_f{floor}_r{row}_c{col}"
            create_face_shape(ctx, name=wname, shape="rectangle",
                              width=WINDOW_W, length=WINDOW_H,
                              direction=[0, 1, 0])
            wx = col * ROOM_W + ROOM_W / 2
            wy = row * ROOM_L
            wz = floor * ROOM_H + 1.5
            pname = f"{wname}_placed"
            transform(ctx, name=pname, topology_name=wname,
                      operation="translate", x=wx, y=wy, z=wz)
            window_names.append(pname)
print(f"  → {len(window_names)} windows")

# ══════════════════════════════════════════════════════════════
# 5. Create doors between horizontally adjacent rooms (per floor)
# ══════════════════════════════════════════════════════════════
print("Creating doors...")
door_names = []
door_id = 0
for floor in range(FLOORS):
    for row in range(ROWS):
        for col in range(COLS):
            # Door to right neighbor
            if col < COLS - 1:
                dname = f"door_{door_id}"
                door_id += 1
                create_face_shape(ctx, name=dname, shape="rectangle",
                                  width=DOOR_W, length=DOOR_H,
                                  direction=[1, 0, 0])
                dx = (col + 1) * ROOM_W
                dy = row * ROOM_L + ROOM_L / 2
                dz = floor * ROOM_H + DOOR_H / 2
                pname = f"{dname}_placed"
                transform(ctx, name=pname, topology_name=dname,
                          operation="translate", x=dx, y=dy, z=dz)
                door_names.append(pname)

            # Door to back neighbor
            if row < ROWS - 1:
                dname = f"door_{door_id}"
                door_id += 1
                create_face_shape(ctx, name=dname, shape="rectangle",
                                  width=DOOR_W, length=DOOR_H,
                                  direction=[0, 1, 0])
                dx = col * ROOM_W + ROOM_W / 2
                dy = (row + 1) * ROOM_L
                dz = floor * ROOM_H + DOOR_H / 2
                pname = f"{dname}_placed"
                transform(ctx, name=pname, topology_name=dname,
                          operation="translate", x=dx, y=dy, z=dz)
                door_names.append(pname)

            # Door to floor above (vertical connection through ceiling)
            if floor < FLOORS - 1:
                dname = f"door_{door_id}"
                door_id += 1
                create_face_shape(ctx, name=dname, shape="rectangle",
                                  width=DOOR_W, length=DOOR_W,
                                  direction=[0, 0, 1])
                dx = col * ROOM_W + ROOM_W / 2
                dy = row * ROOM_L + ROOM_L / 2
                dz = (floor + 1) * ROOM_H
                pname = f"{dname}_placed"
                transform(ctx, name=pname, topology_name=dname,
                          operation="translate", x=dx, y=dy, z=dz)
                door_names.append(pname)

print(f"  → {len(door_names)} doors ({door_id} total: horizontal + vertical)")

# ══════════════════════════════════════════════════════════════
# 6. Cluster everything into a full model
# ══════════════════════════════════════════════════════════════
print("Clustering all elements...")
create_cluster(ctx, name="all_windows", topology_names=window_names)
create_cluster(ctx, name="all_doors", topology_names=door_names)
create_cluster(ctx, name="full_model",
               topology_names=["building", "all_windows", "all_doors"])

# ══════════════════════════════════════════════════════════════
# 7. Adjacency graph
# ══════════════════════════════════════════════════════════════
print("Building adjacency graph...")
result = create_graph_from_topology(ctx, name="building_graph",
                                     topology_name="building",
                                     direct=True, via_shared_topologies=True)
print(f"  → {result}")

# ══════════════════════════════════════════════════════════════
# 8. Dictionary metadata
# ══════════════════════════════════════════════════════════════
dictionary(ctx, topology_name="building", action="set", data={
    "building_name": "Test Building",
    "floors": FLOORS,
    "rooms_per_floor": ROWS * COLS,
    "total_rooms": FLOORS * ROWS * COLS,
    "total_windows": len(window_names),
    "total_doors": len(door_names),
})

# ══════════════════════════════════════════════════════════════
# 9. Query & report
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("BUILDING SUMMARY")
print("=" * 60)
print(query_topology(ctx, topology_name="building", detail="summary"))
print(query_topology(ctx, topology_name="full_model", detail="summary"))
print()
meta = dictionary(ctx, topology_name="building", action="get")
print(f"Metadata: {meta}")

subs = get_sub_topologies(ctx, topology_name="building", sub_type="cell")
print(f"Cells: {subs}")
subs = get_sub_topologies(ctx, topology_name="building", sub_type="face")
print(f"Faces: {subs}")

listing = json.loads(list_topologies(ctx))
print(f"\nTotal objects in session: {len(listing)}")

# ══════════════════════════════════════════════════════════════
# 10. Export
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXPORTING")
print("=" * 60)

# Building only
brep_path = os.path.join(OUT_DIR, "building.brep")
print(export_topology(ctx, topology_name="building", file_path=brep_path, format="brep"))

obj_path = os.path.join(OUT_DIR, "building.obj")
print(export_topology(ctx, topology_name="building", file_path=obj_path, format="obj", transpose=True))

# Full model (building + windows + doors)
full_brep = os.path.join(OUT_DIR, "full_model.brep")
print(export_topology(ctx, topology_name="full_model", file_path=full_brep, format="brep"))

full_obj = os.path.join(OUT_DIR, "full_model.obj")
print(export_topology(ctx, topology_name="full_model", file_path=full_obj, format="obj", transpose=True))

print(f"\nAll files in: {OUT_DIR}/")
for f in sorted(os.listdir(OUT_DIR)):
    size = os.path.getsize(os.path.join(OUT_DIR, f))
    print(f"  {f:30s} {size:>10,} bytes")

print("\nDone!")
