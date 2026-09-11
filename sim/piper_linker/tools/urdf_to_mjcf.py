"""Convert the Linker Hand O6 URDF to a MuJoCo MJCF model.

Why not mujoco's built-in URDF loader?  It drops the inertial of the root link
(no joint -> no body) and ignores <mimic> couplings.  This converter instead:

  - keeps the root link as a real body (with freejoint) so the hand can be
    welded onto an arm flange,
  - turns every <mimic> joint into an equality constraint
    (q_child = offset + multiplier * q_parent), matching the linkage
    coupling of the real O6 hand,
  - adds position actuators only on the 6 independently driven joints
    (thumb yaw, thumb pitch, 4x finger MCP pitch), which is the actual
    actuator layout of the O6 (the DIP/IP joints are passively coupled),
  - derives diaginertia + quat from the full URDF inertia tensor.

Usage:
    python tools/urdf_to_mjcf.py linkerhand-urdf/O6/left/linkerhand_o6_left.urdf \
        -o linkerhand_o6_left/linkerhand_o6_left.xml
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

MIMIC_JOINTS = {
    # child joint: (parent joint, multiplier, offset)
}


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF fixed-axis X-Y-Z (roll, pitch, yaw) -> quaternion (w, x, y, z)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    # R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    R = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])
    return mat_to_quat(R)


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> quaternion (w, x, y, z), Shepperd's method."""
    R = np.asarray(R, dtype=float)
    q = np.zeros(4)
    tr = np.trace(R)
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        q[0] = 0.25 * s
        q[1] = (R[2, 1] - R[1, 2]) / s
        q[2] = (R[0, 2] - R[2, 0]) / s
        q[3] = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q[0] = (R[2, 1] - R[1, 2]) / s
        q[1] = 0.25 * s
        q[2] = (R[0, 1] + R[1, 0]) / s
        q[3] = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q[0] = (R[0, 2] - R[2, 0]) / s
        q[1] = (R[0, 1] + R[1, 0]) / s
        q[2] = 0.25 * s
        q[3] = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q[0] = (R[1, 0] - R[0, 1]) / s
        q[1] = (R[0, 2] + R[2, 0]) / s
        q[2] = (R[1, 2] + R[2, 1]) / s
        q[3] = 0.25 * s
    q /= np.linalg.norm(q)
    return q


def inertia_to_diag(ixx, ixy, ixz, iyy, iyz, izz):
    """Full URDF inertia tensor -> (diaginertia, quat) for MJCF."""
    I = np.array([
        [ixx, ixy, ixz],
        [ixy, iyy, iyz],
        [ixz, iyz, izz],
    ])
    evals, evecs = np.linalg.eigh(I)
    # eigh returns ascending; MJCF convention is descending diaginertia.
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    if np.linalg.det(evecs) < 0.0:  # keep the frame right-handed
        evecs[:, 0] *= -1.0
    return evals, mat_to_quat(evecs)


def fmt(vals):
    return " ".join(f"{v:.8g}" for v in vals)


def parse_origin(elem) -> tuple[list[float], list[float]]:
    """Parse <origin xyz= rpy=> of a URDF element (order-independent)."""
    xyz = rpy = None
    o = elem.find("origin")
    if o is not None:
        xyz = o.get("xyz", "0 0 0")
        rpy = o.get("rpy", "0 0 0")
    return ([float(v) for v in xyz.split()],
            [float(v) for v in rpy.split()])


def convert(urdf_path: Path, out_path: Path, meshdir: str = "meshes") -> None:
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # ---- links ---------------------------------------------------------
    links = {}
    for link in root.findall("link"):
        name = link.get("name")
        links[name] = link

    # ---- materials -----------------------------------------------------
    materials = {}
    for mat in root.findall("material"):
        color = mat.find("color")
        if color is not None and color.get("rgba"):
            materials[mat.get("name")] = color.get("rgba")

    # ---- joint tree ----------------------------------------------------
    joints = {}
    children = {}  # child link name -> (parent link name, joint element)
    for joint in root.findall("joint"):
        jname = joint.get("name")
        jtype = joint.get("type")
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        if jtype != "fixed":
            joints[jname] = joint
        children[child] = (parent, joint)

    root_link = next(l for l in links if l not in children)

    children_by_parent: dict[str, list[str]] = {}
    for child, (parent, _) in children.items():
        children_by_parent.setdefault(parent, []).append(child)

    order: list[str] = []

    def visit(name: str) -> None:
        order.append(name)
        for c in children_by_parent.get(name, []):
            visit(c)

    visit(root_link)
    assert len(order) == len(links), "link tree has unreachable links"

    mimic = {}       # child joint -> (parent joint, multiplier, offset)
    actuators = []   # joints with independent actuators
    for jname, joint in joints.items():
        m = joint.find("mimic")
        if m is not None:
            mimic[jname] = (m.get("joint"),
                            float(m.get("multiplier", "1")),
                            float(m.get("offset", "0")))
        else:
            actuators.append(jname)

    # ---- emit MJCF -----------------------------------------------------
    lines = []
    add = lines.append
    add('<mujoco model="linkerhand_o6_left">')
    add(f'  <compiler angle="radian" meshdir="{meshdir}"/>')
    add('  <option integrator="implicitfast" cone="elliptic"/>')
    add("")
    add("  <default>")
    add('    <default class="hand">')
    add('      <joint damping="0.05" frictionloss="0.15" armature="0.001"/>')
    add('      <position kp="30" kv="2" forcerange="-100 100"/>')
    add('      <default class="visual">')
    add('        <geom group="2" type="mesh" contype="0" conaffinity="0" density="0"/>')
    add("      </default>")
    add('      <default class="collision">')
    add('        <geom group="3" type="mesh"/>')
    add("      </default>")
    add("    </default>")
    add("  </default>")
    add("")
    add("  <asset>")
    for mname, rgba in materials.items():
        add(f'    <material name="{mname}" rgba="{rgba}"/>')
    for lname in links:
        link = links[lname]
        mesh = None
        for kind in ("visual", "collision"):
            for elem in link.findall(kind):
                m = elem.find("geometry").find("mesh")
                if m is not None:
                    mesh = m.get("filename").split("/")[-1]
                    break
            if mesh:
                break
        stem = mesh.replace(".STL", "").replace(".stl", "")
        add(f'    <mesh name="{stem}" file="{mesh}"/>')  # meshdir is prepended
    add("  </asset>")
    add("")
    add("  <worldbody>")

    def emit_body(lname: str, indent: int, is_root: bool) -> None:
        pad = "  " * indent
        link = links[lname]
        if is_root:
            # root link: model frame == link frame, freejoint for later welding
            add(f"{pad}<body name=\"{lname}\" childclass=\"hand\">")
            add(f"{pad}  <freejoint/>")
        else:
            # body frame == joint frame: pos/quat from the joint origin
            _, joint = children[lname]
            jxyz, jrpy = parse_origin(joint)
            add(f"{pad}<body name=\"{lname}\" pos=\"{fmt(jxyz)}\" quat=\"{fmt(rpy_to_quat(*jrpy))}\">")

        inertial = link.find("inertial")
        ixyz, _ = parse_origin(inertial)
        mass = float(inertial.find("mass").get("value"))
        inertia = inertial.find("inertia")
        diag, iquat = inertia_to_diag(
            float(inertia.get("ixx")), float(inertia.get("ixy")),
            float(inertia.get("ixz")), float(inertia.get("iyy")),
            float(inertia.get("iyz")), float(inertia.get("izz")))
        add(f"{pad}  <inertial pos=\"{fmt(ixyz)}\" quat=\"{fmt(iquat)}\" "
            f"mass=\"{mass:.8g}\" diaginertia=\"{fmt(diag)}\"/>")

        for vis in link.findall("visual"):
            vxyz, vrpy = parse_origin(vis)
            geo = vis.find("geometry")
            mesh = geo.find("mesh")
            assert mesh is not None, "only mesh visuals are supported"
            mname = (mesh.get("filename").split("/")[-1]
                     .replace(".STL", "").replace(".stl", ""))
            mat = vis.find("material")
            matref = f' material="{mat.get("name")}"' if mat is not None and mat.get("name") else ""
            add(f'{pad}  <geom class="visual" pos="{fmt(vxyz)}" quat="{fmt(rpy_to_quat(*vrpy))}" '
                f'mesh="{mname}"{matref}/>')

        for col in link.findall("collision"):
            cxyz, crpy = parse_origin(col)
            geo = col.find("geometry")
            mesh = geo.find("mesh")
            assert mesh is not None, "only mesh collisions are supported"
            mname = (mesh.get("filename").split("/")[-1]
                     .replace(".STL", "").replace(".stl", ""))
            add(f'{pad}  <geom class="collision" pos="{fmt(cxyz)}" quat="{fmt(rpy_to_quat(*crpy))}" '
                f'mesh="{mname}"/>')

        # joint (for non-root links): the joint frame IS the body frame
        if not is_root:
            _, joint = children[lname]
            jname = joint.get("name")
            axis = joint.find("axis")
            axis_str = axis.get("xyz") if axis is not None else "0 0 1"
            limit = joint.find("limit")
            lo = float(limit.get("lower", "0"))
            hi = float(limit.get("upper", "0"))
            effort = float(limit.get("effort", "100"))
            if jname in mimic:
                parent, mult, off = mimic[jname]
                # widen the child range to what the coupling can reach
                pjoint = joints[parent]
                plimit = pjoint.find("limit")
                plo = float(plimit.get("lower", "0"))
                phi = float(plimit.get("upper", "0"))
                lo = off + mult * plo
                hi = off + mult * phi
            add(f'{pad}  <joint name="{jname}" '
                f'axis="{axis_str}" range="{lo:.8g} {hi:.8g}" '
                f'actuatorfrcrange="-{effort:.8g} {effort:.8g}"/>')

        for child in children_by_parent.get(lname, []):
            emit_body(child, indent + 1, False)
        add(f"{pad}</body>")

    emit_body(root_link, 2, True)
    add("  </worldbody>")
    add("")

    if mimic:
        add("  <equality>")
        for child, (parent, mult, off) in mimic.items():
            add(f'    <joint joint1="{child}" joint2="{parent}" '
                f'polycoef="{off:.8g} {mult:.8g} 0 0 0"/>')
        add("  </equality>")
        add("")

    add("  <actuator>")
    for jname in actuators:
        add(f'    <position name="{jname}" joint="{jname}" class="hand"/>')
    add("  </actuator>")
    add("")
    add("</mujoco>")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")

    print(f"wrote {out_path}")
    print(f"  links: {len(links)}, joints: {len(joints)}")
    print(f"  independent (actuated): {sorted(actuators)}")
    print(f"  mimic (equality-coupled): {mimic}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("urdf")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()
    convert(Path(args.urdf), Path(args.output))


if __name__ == "__main__":
    main()
