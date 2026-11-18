"""YAM linear_4310 gripper meshes in the gripper frame (mm): y = jaw slide axis, fingers point -z,
+x = handle side (opposite the D405 wrist camera, which sits at -x)."""
import numpy as np, trimesh
from trimesh.transformations import quaternion_matrix, translation_matrix
def TM(p, q): M = quaternion_matrix(q); M[:3, 3] = p; return M
class Yam:
    def __init__(s, RM):
        s.A = f"{RM}/gripper/linear_4310/assets"; s.RM = RM
        s.body = trimesh.load(f"{s.A}/gripper.stl"); s.body.apply_transform(TM([-0.014, -0.0463995, 0.0731], [1, 0, 0, 0])); s.body.apply_scale(1000)
        A = TM([-0.0463995, -0.014, -0.0731], [0, 0.7071067811865476, 0.7071067811865476, 0])
        st2mine = TM([-0.014, -0.0463995, 0.0731], [1, 0, 0, 0]) @ np.linalg.inv(A)
        C = TM([-7.656797949839866e-08, -0.07043497874827391, 0.07700592901799347], [-0.9762948880484529, 0.2164446616738001, 1.7930671418198152e-06, 3.9751623626127497e-07])
        s.cam = trimesh.load(f"{RM}/station/assets/d405_wrist_linear_4310.stl"); s.cam.apply_transform(st2mine @ C); s.cam.apply_scale(1000)
        s._tip = {n: trimesh.load(f"{s.A}/{n}.stl") for n in ("tip_left", "tip_right")}
    def tip(s, name, op):
        """whole jaw (carriage + finger + stock screws) at opening op mm per jaw; tip_left = jaw L (-y), tip_right = jaw R (+y)"""
        m = s._tip[name].copy()
        if name == "tip_left":
            m.apply_transform(TM([0.129783, 0.00999321, -0.0914614], [0.499998, 0.5, 0.500002, 0.5])); m.apply_transform(translation_matrix([0, 0, -op / 1000])); m.apply_transform(TM([-0.0238981, 0.0450619, -0.0545599], [0.499998, -0.5, -0.5, -0.500002]))
        else:
            m.apply_transform(TM([-0.0379932, 0.129783, 0.00133753], [0.707105, -0.707108, 0, 0])); m.apply_transform(translation_matrix([0, 0, -op / 1000])); m.apply_transform(TM([0.0238981, -0.0450619, -0.0545599], [0.707105, 0.707108, 0, 0]))
        m.apply_scale(1000); return m
    def split(s, name, op, shift=0.0, zcut=-70.45):
        """(carriage, finger) with the finger moved down by shift (the spacer thickness); stock screw bodies dropped"""
        m = max(s.tip(name, op).split(only_watertight=False), key=lambda c: len(c.faces))
        car = trimesh.intersections.slice_mesh_plane(m, [0, 0, 1], [0, 0, zcut]); fin = trimesh.intersections.slice_mesh_plane(m, [0, 0, -1], [0, 0, zcut])
        fin.apply_translation([0, 0, -shift]); return car, fin
