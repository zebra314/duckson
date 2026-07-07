import logging
from pathlib import Path
import genesis as gs
import torch

from base.robot import Robot


class Duckson(Robot):
    def __init__(self):
        super().__init__()

        self.logger = logging.getLogger(__name__)

        sim_root = Path(__file__).parent
        self.model_path = sim_root / "xmls" / "open_duck_mini_v2.xml"

        self.num_envs = 3
        self.env_spacing = (1.0, 1.0)

        self.motor_names = [
            "left_hip_yaw",
            "left_hip_roll",
            "left_hip_pitch",
            "left_knee",
            "left_ankle",
            "right_hip_yaw",
            "right_hip_roll",
            "right_hip_pitch",
            "right_knee",
            "right_ankle",
            "neck_pitch",
            "head_pitch",
            "head_yaw",
            "head_roll",
        ]
        self.num_motors = len(self.motor_names)

        self.entity = None
        self.motor_dof_indices = None

    def build_entity(self, scene):
        # Add entity
        morph = gs.morphs.MJCF(file=self.model_path)
        self.entity = scene.add_entity(morph)

        # Build dof indices
        all_dof_names = [joint.name for joint in self.entity.joints]
        self.motor_dof_indices = torch.tensor(
            [all_dof_names.index(name) for name in self.motor_names],
            device=gs.device,
        )

        scene.build(n_envs=self.num_envs, env_spacing=self.env_spacing)

    def set_dof_properties(self):
        self.entity.set_dofs_kp(
            kp=torch.tensor([17.8] * self.num_motors, device=gs.device),
            dofs_idx_local=self.motor_dof_indices,
        )

        self.entity.set_dofs_kv(
            kv=torch.tensor([0.0] * self.num_motors, device=gs.device),
            dofs_idx_local=self.motor_dof_indices,
        )

        self.entity.set_dofs_force_range(
            lower=torch.tensor(
                [-3.35] * self.num_motors, device=gs.device
            ),
            upper=torch.tensor([3.35] * self.num_motors, device=gs.device),
            dofs_idx_local=self.motor_dof_indices,
        )

    def apply_command(self, command: list):
        cmd = command[0]

        action = cmd.get("action")
        params = cmd.get("params")

        t = params.get("t")
        actions = torch.zeros((self.num_envs, self.num_motors), device=gs.device)

        actions[0, 5] = torch.sin(torch.tensor(t * 10.0)) * 0.5
        actions[0, 10] = torch.sin(torch.tensor(t * 10.0)) * 0.5

        self.entity.control_dofs_position(
            actions, self.motor_dof_indices
        )

    def print_robot_info(self):
        if self.entity is None:
            raise ValueError("Entity is not built yet. Please call build() first.")

        print(self.entity.joints)
