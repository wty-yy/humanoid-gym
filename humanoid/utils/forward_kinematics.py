from collections import defaultdict
import os
from pathlib import Path

from isaacgym.torch_utils import quat_from_angle_axis, quat_mul, quat_rotate
from isaacgym import gymtorch, gymapi, gymutil
import torch
from torch import nn
import mujoco
import numpy as np

from humanoid import LEGGED_GYM_ROOT_DIR

KUAVO_S42_XML_PATH = os.path.join(LEGGED_GYM_ROOT_DIR, "resources", "robots", "kuavo_s42", "mjcf", "biped_s42.xml")
KUAVO_S42_FINE_FIXED_ARM_XML_PATH = str(Path(LEGGED_GYM_ROOT_DIR) / "resources/robots/biped_s42_fine/xml/biped_s42_only_lower_body.xml")


class ForwardKinematics(nn.Module):
    def __init__(self, mjcf_path=KUAVO_S42_FINE_FIXED_ARM_XML_PATH):
        super().__init__()
        self.body_info = {}
        self.root_body = ""

        self.parse_mujoco(mjcf_path)

        self.free_body_num = len(self.body_info)
        self.body_pos_tensor = nn.Parameter(torch.zeros(self.free_body_num, 3), requires_grad=False)
        self.body_axis_tensor = nn.Parameter(torch.zeros(self.free_body_num, 3), requires_grad=False)
        self.body_range_tensor = nn.Parameter(torch.zeros(self.free_body_num, 2), requires_grad=False)

        for body_name, info in self.body_info.items():
            self.body_pos_tensor.data[info["idx"]] = torch.Tensor(info["pos"])
            self.body_axis_tensor.data[info["idx"]] = torch.Tensor(info["axis"])
            self.body_range_tensor.data[info["idx"]] = torch.Tensor(info["range"])

    def parse_mujoco(self, file_path):
        model = mujoco.MjModel.from_xml_path(file_path)

        assert sum(model.body_parentid == 0) == 2

        for body_id in range(model.nbody):
            body = model.body(body_id)
            if body_id == 0: # skip world_body
                continue
            if body.parentid == 0: # skip base_link
                self.root_body = body.name
                continue

            assert np.allclose(body.quat, [1, 0, 0, 0]) # body rotation not support yet
            if body.jntnum == 0: continue
            assert body.jntnum == 1
            self.body_info[body.name] = {
                "parent": model.body(body.parentid).name,
                "pos": body.pos,
            }

        for joint_id in range(model.njnt):
            joint = model.joint(joint_id)
            body_name = model.body(joint.bodyid[0]).name
            if body_name not in self.body_info:
                continue

            assert np.allclose(joint.pos, 0)
            assert joint.type == 3 # support hinge only now

            self.body_info[body_name]["joint_type"] = "hinge"
            self.body_info[body_name]["axis"] = joint.axis
            self.body_info[body_name]["range"] = joint.range
            self.body_info[body_name]["idx"] = joint.qposadr[0] - 7


    def forward(self, qpos, with_root=False):
        """

        :param qpos: data of pos, frames * dof_num if with_root==False or frames * (7 + dof_num)
            if with_root==True
        :param with_root: whether dof_pos including root data
        :return: abstract point in dictionary
        """
        frames = qpos.shape[0] # dof_pos: frames * dof_num or frames * (7 + dof_num)
        device = qpos.device
        body_positions = {}
        body_rotations = {}

        if with_root:
            body_positions[self.root_body] = qpos[:, :3]
            body_rotations[self.root_body] = qpos[:, 3:7]
            dof_pos = torch.clip(qpos[:, 7:], self.body_range_tensor[:, 0], self.body_range_tensor[:, 1])
        else:
            body_positions[self.root_body] = torch.zeros((frames, 3), device=device)
            body_rotations[self.root_body] = torch.tensor([[0., 0., 0., 1.]], device=device).repeat((frames, 1))
            dof_pos = torch.clip(qpos, self.body_range_tensor[:, 0], self.body_range_tensor[:, 1])

        def compute_body_transform(body_name):
            if body_name in body_positions:
                return body_positions[body_name], body_rotations[body_name]
            idx = self.body_info[body_name]["idx"]

            parent_body = self.body_info[body_name]['parent']
            parent_pos, parent_rot = compute_body_transform(parent_body)

            joint_type = self.body_info[body_name]['joint_type']
            joint_axis = self.body_axis_tensor[idx].reshape(-1, 3).repeat((frames, 1))
            joint_angle = dof_pos[:, idx]

            if joint_type == 'hinge':
                joint_rot = quat_from_angle_axis(joint_angle, joint_axis)
                body_rot = quat_mul(parent_rot, joint_rot)
            else:
                raise ValueError(f"Unsupported joint type: {joint_type}")

            relative_body_pos = self.body_pos_tensor[idx].reshape(-1, 3).repeat((frames, 1))
            body_pos = parent_pos + quat_rotate(parent_rot, relative_body_pos)

            body_positions[body_name] = body_pos.clone()
            body_rotations[body_name] = body_rot.clone()

        for body_name in self.body_info:
            if body_name != self.root_body:
                compute_body_transform(body_name)

        return body_positions, body_rotations

if __name__ == '__main__':
    fk = ForwardKinematics()
    print(fk.free_body_num)
    qpos = torch.rand(1024, fk.free_body_num)
    body_positions, body_rotations = fk(qpos, with_root=False)

    print(body_positions.keys())
    print(body_positions, body_rotations)
    # print("Test passed!")

