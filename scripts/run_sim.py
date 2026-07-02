import genesis as gs
import torch

from sim.duckson import Duckson


def main():
    gs.init(backend=gs.cpu)

    scene = gs.Scene(show_viewer=True)
    scene.add_entity(gs.morphs.Plane())

    duckson = Duckson()
    duckson.build_entity(scene)

    scene.build(n_envs=4, env_spacing=(1.0, 1.0))

    LEFT_KNEE_IDX = 5
    RIGHT_KNEE_IDX = 10

    for i in range(1000):
        t = i * 0.01

        actions = torch.zeros((4, 14), device=gs.device)

        actions[0, LEFT_KNEE_IDX] = torch.sin(torch.tensor(t * 10.0)) * 0.5
        actions[0, RIGHT_KNEE_IDX] = torch.sin(torch.tensor(t * 10.0)) * 0.5

        duckson.apply_actions(actions)

        scene.step()

if __name__ == "__main__":
    main()
