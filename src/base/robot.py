class Robot:
    def __init__(self):
        self.entity = None
        self.motor_dof_indices = None

    def build_entity(self, scene):
        raise NotImplementedError("Subclasses must implement the build() method.")

    def apply_actions(self, actions_tensor):
        self.entity.control_dofs_position(
            actions_tensor, self.motor_dof_indices
        )

    def print_robot_info(self):
        if self.entity is None:
            raise ValueError("Entity is not built yet. Please call build() first.")

        print(self.entity.joints)
