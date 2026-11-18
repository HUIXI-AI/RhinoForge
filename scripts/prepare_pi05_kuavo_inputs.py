#!/usr/bin/env python3
"""Prepare public Kuavo inputs for examples/configs/pi05/kuavo/*.toml (CPU only).

Real robot source: the Apache-2.0 LeRobot v2.1 dataset RoboCOIN/Leju_Kuavo_4_box_storage_parcel
(three 848x480 RGB cameras: head, left wrist, right wrist). Download it first, e.g.
``huggingface-cli download RoboCOIN/Leju_Kuavo_4_box_storage_parcel --repo-type dataset --local-dir <root>``.

Simulation source (optional ``--sim-bag``): one ROS1 bag of Leju's ICRA 2026 challenge
dataset LejuRobotics/kuavo_data_challenge_icra, ``sim/TASK2-ParcelWeighing`` (Kuavo 4 Pro in
simulation, three 640x480 JPEG cameras on the kuavo_deploy topics, Robotiq 2F-85
grippers; the dataset card states no license). Reading it needs ``pip install rosbags``.

Writes into ``--out``:
  frames/cam_h.png, cam_l.png, cam_r.png      one 848x480 real frame per camera (decoded video frame)
  frames_sim/cam_h.png, cam_l.png, cam_r.png  one 640x480 simulation frame per camera (with --sim-bag)
  observation.json                            physical state per robot layout, for [input].state
  norm_stats/<layout>/norm_stats.json         openpi quantile statistics (state + 50-step actions)
  report.json                                 dataset identity and derivation settings

The state follows the LeTools KuavoRosbag2Lerobot layout: [left arm 7 (rad), left
hand 1, right arm 7 (rad), right hand 1] (+ 4 lower-body joints for the 5-W whole
body). LeTools scales the end effector to 0..1 in conversion and deployment
(is_binary=false: dexterous hand and claw /100, rq2f85 gripper /0.8); RoboCOIN stores
the hand position in radians of the 0..100 command, so the hand is scaled by
180/pi/100. Action statistics use openpi's training transform: 10 Hz chunks of 50
steps, arm joints as deltas from the current state (make_bool_mask(7, -1, 7, -1,
lower)), hands absolute, q01/q99 from openpi RunningStats' 5000-bin histograms. The
dataset has no lower-body joints: the whole-body layout uses zero state and identity
statistics (q01=-1, q99=1) for those 4 dims, a placeholder rather than robot data.
The simulation sample uses the same layouts and statistics: arms from
``/sensors_data_raw`` joint_q[12:26] (4 Pro), grippers from ``/gripper/state`` /0.8.

  python scripts/prepare_pi05_kuavo_inputs.py --dataset <root> --out <dir> [--episode 0 --frame 300]
      [--sim-bag <task2_0001.bag> --sim-frame 200]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SOURCE = "RoboCOIN/Leju_Kuavo_4_box_storage_parcel (Apache-2.0, LeRobot v2.1)"
CAMERAS = {"cam_h": "cam_front_head_rgb", "cam_l": "cam_left_wrist_rgb", "cam_r": "cam_right_wrist_rgb"}
HORIZON, SOURCE_FPS, TRAIN_HZ, LOWER_BODY = 50, 30, 10, 4
SIM_SOURCE = ("LejuRobotics/kuavo_data_challenge_icra sim/TASK2-ParcelWeighing "
              "(Leju ICRA 2026 challenge, ROS1 bag; no license stated)")
SIM_CAMERAS = {"cam_h": "/cam_h/color/image_raw/compressed", "cam_l": "/cam_l/color/image_raw/compressed",
               "cam_r": "/cam_r/color/image_raw/compressed"}
SIM_FRAME_WH = (640, 480)
ARM_4PRO = (12, 26)  # LeTools configs/platform/platform_config.yaml: 4pro joint_q[12:26]
RQ2F85_SCALE = 0.8   # LeTools obs_buffer rq2f85State_callback / CvtRosbag2Lerobot: position / 0.8


def _side(side: str) -> list[str]:
    return [f"{side}_arm_joint_{index}_rad" for index in range(1, 8)] + [f"{side}_hand_joint_1_rad"]


# layout: (dataset columns, lower-body dims, delta mask without the lower body)
LAYOUTS = {
    "single_arm": (_side("right"), 0, [True] * 7 + [False]),
    "dual_arm": (_side("left") + _side("right"), 0, ([True] * 7 + [False]) * 2),
    "wholebody": (_side("left") + _side("right"), LOWER_BODY, ([True] * 7 + [False]) * 2),
}


def delta_mask(layout: str) -> list[bool]:
    _, lower, mask = LAYOUTS[layout]
    return mask + [True] * lower


def _names(info: dict, key: str) -> list[str]:
    names = info["features"][key]["names"]
    return list(names.values())[0] if isinstance(names, dict) else list(names)


def _to_letools(values, names):
    """RoboCOIN hand radians -> LeTools 0..1 (the 0..100 command / 100; in place, float32)."""
    import numpy as np

    for side in ("left", "right"):
        values[..., names.index(f"{side}_hand_joint_1_rad")] *= np.float32(180.0 / np.pi / 100.0)
    return values


def _layout_states(left_arm, left_eef, right_arm, right_eef) -> dict:
    """Physical state per layout in the LeTools order; shortest float32 text per value."""
    import numpy as np

    sides = {"left": list(left_arm) + [left_eef], "right": list(right_arm) + [right_eef]}
    states = {}
    for layout, (columns, lower, _) in LAYOUTS.items():
        values = [value for side in ("left", "right") if any(name.startswith(side) for name in columns)
                  for value in sides[side]] + [0.0] * lower
        # The TOML value parses back to the same float32.
        states[layout] = [float(str(np.float32(value))) for value in values]
    return states


def running_stats(values) -> dict:
    """openpi RunningStats after one update: mean/std and histogram q01/q99."""
    import numpy as np

    values = values.reshape(-1, values.shape[-1]).astype(np.float64)
    mean = values.mean(0)
    stats = {"mean": mean, "std": np.sqrt(np.maximum(0.0, (values ** 2).mean(0) - mean ** 2))}
    for name, fraction in (("q01", 0.01), ("q99", 0.99)):
        result = []
        for column in values.T:
            edges = np.linspace(column.min() - 1e-10, column.max() + 1e-10, 5001)
            counts, _ = np.histogram(column, bins=edges)
            result.append(edges[np.searchsorted(np.cumsum(counts), fraction * len(column))])
        stats[name] = np.asarray(result)
    return stats


def _episodes(root: Path, info: dict):
    import numpy as np
    import pyarrow.parquet as pq

    state_names, action_names = _names(info, "observation.state"), _names(info, "action")
    for path in sorted((root / "data").rglob("episode_*.parquet")):
        table = pq.read_table(path, columns=["observation.state", "action"])
        state = np.stack(table.column("observation.state").to_numpy(zero_copy_only=False)).astype(np.float32)
        action = np.stack(table.column("action").to_numpy(zero_copy_only=False)).astype(np.float32)
        yield _to_letools(state, state_names), _to_letools(action, action_names)


def norm_stats(root: Path, info: dict) -> dict:
    """{layout: {"state": stats, "actions": stats}} over every episode."""
    import numpy as np

    state_names, action_names = _names(info, "observation.state"), _names(info, "action")
    episodes = list(_episodes(root, info))
    step = SOURCE_FPS // TRAIN_HZ
    result = {}
    for layout, (columns, lower, mask) in LAYOUTS.items():
        if layout == "wholebody":
            continue
        state_index = [state_names.index(name) for name in columns]
        action_index = [action_names.index(name) for name in columns]
        flags = np.asarray(mask)
        states, chunks = [], []
        for state, action in episodes:
            for phase in range(step):
                current, target = state[phase::step][:, state_index], action[phase::step][:, action_index]
                if len(current) < 2:
                    continue
                rows = np.minimum(np.arange(len(current))[:, None] + np.arange(HORIZON)[None, :], len(current) - 1)
                chunk = target[rows]
                chunk[:, :, flags] -= current[:, None, flags]
                states.append(current)
                chunks.append(chunk)
        result[layout] = {"state": running_stats(np.concatenate(states)),
                          "actions": running_stats(np.concatenate(chunks))}
    identity = {"mean": np.zeros(LOWER_BODY), "std": np.ones(LOWER_BODY),
                "q01": -np.ones(LOWER_BODY), "q99": np.ones(LOWER_BODY)}
    result["wholebody"] = {key: {name: np.concatenate([value[name], identity[name]]) for name in identity}
                           for key, value in result["dual_arm"].items()}
    return result


def sample(root: Path, info: dict, episode: int, frame: int, out: Path) -> dict:
    """Write one frame per camera and return the physical state per layout."""
    import av
    import numpy as np
    import pyarrow.parquet as pq
    from PIL import Image

    (out / "frames").mkdir(parents=True, exist_ok=True)
    chunk = episode // int(info["chunks_size"])
    shapes = {}
    for key, camera in CAMERAS.items():
        path = root / info["video_path"].format(episode_chunk=chunk, video_key=f"observation.images.{camera}",
                                                episode_index=episode)
        with av.open(str(path)) as container:
            image = next((decoded.to_ndarray(format="rgb24") for index, decoded in
                          enumerate(container.decode(video=0)) if index == frame), None)
        if image is None:
            raise ValueError(f"{path} has no frame {frame}")
        Image.fromarray(image).save(out / "frames" / f"{key}.png")
        shapes[key] = list(image.shape)
    data = root / info["data_path"].format(episode_chunk=chunk, episode_index=episode)
    table = pq.read_table(data, columns=["observation.state", "frame_index"])
    rows = np.nonzero(table.column("frame_index").to_numpy() == frame)[0]
    if len(rows) != 1:
        raise ValueError(f"{data} has no unique frame_index {frame}")
    names = _names(info, "observation.state")
    state = _to_letools(np.asarray(table.column("observation.state")[int(rows[0])].as_py(), dtype=np.float32), names)
    value = {name: state[names.index(name)] for name in names}
    states = _layout_states(*(item for side in ("left", "right") for item in (
        [value[f"{side}_arm_joint_{index}_rad"] for index in range(1, 8)], value[f"{side}_hand_joint_1_rad"])))
    return {"episode": episode, "frame": frame, "image_shapes": shapes, "state": states}


def sim_sample(bag: Path, frame: int, out: Path) -> dict:
    """Write the 640x480 simulation frames at head-camera message ``frame`` and the matching state.

    The wrist frames, joint state and gripper state are the messages whose header
    stamps are nearest to the head frame's (kuavo_deploy aligns its buffers the same way).
    """
    import io

    import numpy as np
    from PIL import Image
    try:
        from rosbags.highlevel import AnyReader
    except ImportError as error:
        raise SystemExit("--sim-bag needs the rosbags package (pip install rosbags)") from error

    topics = {*SIM_CAMERAS.values(), "/sensors_data_raw", "/gripper/state"}
    messages = {topic: [] for topic in topics}
    with AnyReader([bag]) as reader:
        connections = [conn for conn in reader.connections if conn.topic in topics]
        if {conn.topic for conn in connections} != topics:
            raise SystemExit(f"{bag} lacks {sorted(topics - {conn.topic for conn in connections})}")
        for conn, _, raw in reader.messages(connections=connections):
            message = reader.deserialize(raw, conn.msgtype)
            stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
            if conn.topic in SIM_CAMERAS.values():
                payload = bytes(message.data)
            elif conn.topic == "/sensors_data_raw":
                payload = np.asarray(message.joint_data.joint_q, dtype=np.float64)
            else:
                payload = np.asarray(message.position, dtype=np.float64)
            messages[conn.topic].append((stamp, payload))
    head = messages[SIM_CAMERAS["cam_h"]]
    if not 0 <= frame < len(head):
        raise SystemExit(f"{bag} has {len(head)} head-camera frames; --sim-frame {frame} is out of range")
    target = head[frame][0]

    def nearest(topic):
        stamps = np.asarray([stamp for stamp, _ in messages[topic]])
        index = int(np.argmin(np.abs(stamps - target)))
        return messages[topic][index][0] - target, messages[topic][index][1]

    (out / "frames_sim").mkdir(parents=True, exist_ok=True)
    offsets, shapes = {}, {}
    for key, topic in SIM_CAMERAS.items():
        offset, data = nearest(topic)
        with Image.open(io.BytesIO(data)) as image:
            rgb = np.asarray(image.convert("RGB"))
        if rgb.shape != (SIM_FRAME_WH[1], SIM_FRAME_WH[0], 3):
            raise SystemExit(f"{topic} frame is {list(rgb.shape)}, expected {SIM_FRAME_WH[0]}x{SIM_FRAME_WH[1]}")
        # PNG keeps the decoded pixels fixed (JPEG decoders may differ by a level).
        Image.fromarray(rgb).save(out / "frames_sim" / f"{key}.png")
        offsets[key], shapes[key] = round(offset * 1000, 3), list(rgb.shape)
    joint_offset, joint_q = nearest("/sensors_data_raw")
    gripper_offset, gripper = nearest("/gripper/state")
    offsets.update(joint_q=round(joint_offset * 1000, 3), gripper=round(gripper_offset * 1000, 3))
    arms = joint_q[ARM_4PRO[0]:ARM_4PRO[1]]
    grippers = gripper / RQ2F85_SCALE
    states = _layout_states(arms[:7], grippers[0], arms[7:], grippers[1])
    return {"bag": bag.name, "bag_sha256": hashlib.sha256(bag.read_bytes()).hexdigest(), "head_frame": frame,
            "image_shapes": shapes, "stamp_offsets_ms": offsets, "state": states}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True, help="local RoboCOIN Kuavo parcel dataset root")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=300)
    parser.add_argument("--sim-bag", type=Path, help="Kuavo simulation ROS1 bag (kuavo_data_challenge_icra sim)")
    parser.add_argument("--sim-frame", type=int, default=200, help="head-camera message index in --sim-bag")
    args = parser.parse_args(argv)
    info_path = args.dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v2.1" or info.get("fps") != SOURCE_FPS:
        raise SystemExit(f"{args.dataset} is not the expected LeRobot v2.1 {SOURCE_FPS} fps dataset")
    args.out.mkdir(parents=True, exist_ok=True)
    observation = sample(args.dataset, info, args.episode, args.frame, args.out)
    observation.update(source=SOURCE, camera_map={
        "base_0_rgb": "cam_h (cam_front_head_rgb)", "left_wrist_0_rgb": "cam_l (cam_left_wrist_rgb)",
        "right_wrist_0_rgb": "cam_r (cam_right_wrist_rgb)"},
        delta_action_mask={layout: delta_mask(layout) for layout in LAYOUTS})
    if args.sim_bag is not None:
        observation["sim"] = {"source": SIM_SOURCE, **sim_sample(args.sim_bag, args.sim_frame, args.out)}
    (args.out / "observation.json").write_text(json.dumps(observation, indent=2) + "\n", encoding="utf-8")
    for layout, stats in norm_stats(args.dataset, info).items():
        document = {"norm_stats": {key: {name: [float(x) for x in value[name]] for name in ("mean", "std", "q01", "q99")}
                                   for key, value in stats.items()}}
        (args.out / "norm_stats" / layout).mkdir(parents=True, exist_ok=True)
        (args.out / "norm_stats" / layout / "norm_stats.json").write_text(
            json.dumps(document, indent=2) + "\n", encoding="utf-8")
    report = {"source": SOURCE, "info_json_sha256": hashlib.sha256(info_path.read_bytes()).hexdigest(),
              "episodes": info["total_episodes"], "frames": info["total_frames"], "train_hz": TRAIN_HZ,
              "horizon": HORIZON, "lower_body": "no public data: zero state, identity statistics",
              "end_effector": "LeTools 0..1 scale (RoboCOIN hand rad * 180/pi / 100)"}
    if args.sim_bag is not None:
        report["sim"] = {"source": SIM_SOURCE, "bag": args.sim_bag.name, "bag_sha256": observation["sim"]["bag_sha256"],
                         "head_frame": args.sim_frame}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "state": observation["state"],
                      "sim_state": (observation.get("sim") or {}).get("state")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
