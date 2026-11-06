"""NAVSIM v2 agent that replays trajectories saved by eval_action_policy.py (validation/<token>.npz, key 'absolute', shape [8,3],
poses in the ego frame at the scored frame, 0.5 s spacing). No model is run: scoring a saved policy output under the v2 scorer.
Import only inside the navsim2 environment. The token is read from the Scene, so requires_scene must be True."""
import glob
from pathlib import Path

import numpy as np
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SensorConfig, Trajectory


class NpzTrajectoryAgent(AbstractAgent):
    requires_scene = True

    def __init__(self, npz_glob: str, trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5)):
        super().__init__(trajectory_sampling, requires_scene=True)
        self._glob = npz_glob
        self._index = {}

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        self._index = {Path(p).stem: p for p in glob.glob(self._glob)}
        if not self._index:
            raise FileNotFoundError(f"no npz files match {self._glob}")

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_no_sensors()

    def _candidate_tokens(self, scene):
        meta = scene.scene_metadata
        out = []
        frames = getattr(scene, "frames", None)
        n_hist = getattr(meta, "num_history_frames", None)
        if frames is not None and n_hist is not None and 0 < n_hist <= len(frames):
            out.append(frames[n_hist - 1].token)
        out += [getattr(meta, "scene_token", None), getattr(meta, "initial_token", None)]
        return [t for t in out if t]

    def compute_trajectory(self, agent_input, scene=None) -> Trajectory:
        for token in self._candidate_tokens(scene):
            if token in self._index:
                with np.load(self._index[token], allow_pickle=False) as z:
                    poses = np.asarray(z["absolute"], dtype=np.float32)
                return Trajectory(poses.reshape(self._trajectory_sampling.num_poses, 3), self._trajectory_sampling)
        raise KeyError(f"no saved trajectory for any of {self._candidate_tokens(scene)}")
