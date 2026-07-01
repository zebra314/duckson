import time
from pathlib import Path
import mujoco
import mujoco.viewer


def main():
    project_root = Path(__file__).parent.parent
    xml_path = project_root / "src" / "sim" / "xmls" / "open_duck_mini_v2.xml"

    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found at: {xml_path}")

    print(f"Loading MuJoCo model from: {xml_path}")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
