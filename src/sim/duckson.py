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

