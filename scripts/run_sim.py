import time
from pathlib import Path
import mujoco
import mujoco.viewer

def main():
    project_root = Path(__file__).parent.parent

    model_path = project_root / "src" / "sim" / "xmls" / "scene_flat_terrain.xml"

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 1.2
        viewer.cam.lookat = [0.0, 0.0, 0.15]

        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
