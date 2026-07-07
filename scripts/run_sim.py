import genesis as gs
import torch

from sim.duckson import Duckson

def main():
    gs.init(backend=gs.cpu)

    scene = gs.Scene(show_viewer=True)
    scene.add_entity(gs.morphs.Plane())

    duckson = Duckson()
    duckson.build_entity(scene)

    while True:
        t = scene.t

        command = f"""[
            {{"action": "test", "params": {{"t": {t:.2f}}}}}
        ]"""

        duckson.get_command(command)

        scene.step()

if __name__ == "__main__":
    main()
