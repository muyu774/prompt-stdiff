"""Isaac simulation wrapper for the Inspire Hand.

The real implementation drives an Inspire Hand asset inside NVIDIA Isaac
Gym (or Isaac Lab). Those runtimes require a CUDA GPU and a large install,
so importing them is deferred and guarded: this module always imports
cleanly, and the GPU dependency is only touched when you actually
instantiate :class:`IsaacInspireHand`.

For development, testing and dry-runs without a GPU, use
:class:`DummyInspireHand`, which records the commands it receives.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..retarget.inspire_retargeter import INSPIRE_ACTUATORS


class InspireHandSim:
    """Common interface for an Inspire Hand simulation backend."""

    num_actuators = len(INSPIRE_ACTUATORS)

    def reset(self) -> None:
        raise NotImplementedError

    def set_actuator_targets(self, normalized: np.ndarray) -> None:
        """Apply 6 actuator targets in ``[0, 1]`` (1 = fully open)."""
        raise NotImplementedError

    def step(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self) -> "InspireHandSim":
        self.reset()
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class DummyInspireHand(InspireHandSim):
    """No-GPU backend that simply records the most recent command.

    Useful for unit tests, CI and validating a retargeting pipeline before
    touching Isaac. The full command history is kept in :attr:`history`.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.last_target: Optional[np.ndarray] = None
        self.history: List[np.ndarray] = []
        self.steps = 0

    def reset(self) -> None:
        self.last_target = np.ones(self.num_actuators, dtype=np.float64)
        self.history.clear()
        self.steps = 0

    def set_actuator_targets(self, normalized: np.ndarray) -> None:
        target = np.clip(np.asarray(normalized, dtype=np.float64), 0.0, 1.0)
        if target.shape[-1] != self.num_actuators:
            raise ValueError(
                f"Expected {self.num_actuators} targets, got {target.shape[-1]}"
            )
        self.last_target = target
        self.history.append(target.copy())
        if self.verbose:
            pretty = ", ".join(
                f"{n}={v:.2f}" for n, v in zip(INSPIRE_ACTUATORS, target)
            )
            print(f"[DummyInspireHand] {pretty}")

    def step(self) -> None:
        self.steps += 1

    def close(self) -> None:
        pass


class IsaacInspireHand(InspireHandSim):
    """Drive an Inspire Hand asset in Isaac Gym.

    Parameters
    ----------
    asset_root, asset_file:
        Location of the Inspire Hand URDF/MJCF (e.g. ``inspire_hand.urdf``).
    dof_mapping:
        Optional list mapping each of the 6 Inspire actuators to one or more
        simulation DOF indices. Inspire URDFs commonly expose more than 6
        joints (coupled finger links); the mapping fans a single actuator
        command out to the coupled DOFs. When ``None`` an identity mapping
        over the first 6 DOFs is used.
    device:
        Isaac compute device, e.g. ``"cuda:0"``.
    """

    def __init__(
        self,
        asset_root: str,
        asset_file: str = "inspire_hand.urdf",
        dof_mapping: Optional[List[List[int]]] = None,
        device: str = "cuda:0",
        headless: bool = False,
        dt: float = 1.0 / 60.0,
    ):
        self.asset_root = asset_root
        self.asset_file = asset_file
        self.dof_mapping = dof_mapping
        self.device = device
        self.headless = headless
        self.dt = dt

        self._gym = None
        self._sim = None
        self._env = None
        self._actor = None
        self._viewer = None
        self._num_dofs = 0
        self._dof_lower = None
        self._dof_upper = None

    # -- lifecycle -------------------------------------------------------------

    def _import_isaac(self):
        try:
            from isaacgym import gymapi  # type: ignore
            return gymapi
        except Exception as exc:  # pragma: no cover - needs GPU runtime
            raise ImportError(
                "Isaac Gym is required for IsaacInspireHand but could not be "
                "imported. Install Isaac Gym (NVIDIA, requires a CUDA GPU) or "
                "use DummyInspireHand for CPU/offline runs."
            ) from exc

    def reset(self) -> None:  # pragma: no cover - needs GPU runtime
        gymapi = self._import_isaac()
        gym = gymapi.acquire_gym()

        sim_params = gymapi.SimParams()
        sim_params.dt = self.dt
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.use_gpu_pipeline = self.device.startswith("cuda")

        compute_device = 0
        if ":" in self.device:
            compute_device = int(self.device.split(":")[1])
        sim = gym.create_sim(
            compute_device, compute_device, gymapi.SIM_PHYSX, sim_params
        )

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset = gym.load_asset(sim, self.asset_root, self.asset_file, asset_options)
        self._num_dofs = gym.get_asset_dof_count(asset)

        dof_props = gym.get_asset_dof_properties(asset)
        self._dof_lower = np.array(dof_props["lower"], dtype=np.float64)
        self._dof_upper = np.array(dof_props["upper"], dtype=np.float64)

        env = gym.create_env(
            sim,
            gymapi.Vec3(-1, -1, 0),
            gymapi.Vec3(1, 1, 1),
            1,
        )
        pose = gymapi.Transform()
        actor = gym.create_actor(env, asset, pose, "inspire_hand", 0, 0)

        viewer = None
        if not self.headless:
            viewer = gym.create_viewer(sim, gymapi.CameraProperties())

        self._gym, self._sim, self._env = gym, sim, env
        self._actor, self._viewer = actor, viewer

    # -- control ---------------------------------------------------------------

    def _expand_targets(self, normalized: np.ndarray) -> np.ndarray:
        """Fan 6 actuator values out to per-DOF position targets (radians)."""
        norm = np.clip(np.asarray(normalized, dtype=np.float64), 0.0, 1.0)
        targets = np.zeros(self._num_dofs, dtype=np.float64)
        mapping = self.dof_mapping or [[i] for i in range(self.num_actuators)]
        for act_idx, dof_indices in enumerate(mapping):
            if act_idx >= norm.shape[0]:
                break
            for dof in dof_indices:
                if dof < self._num_dofs:
                    lo = self._dof_lower[dof]
                    hi = self._dof_upper[dof]
                    # normalized 1 = open -> lower limit; 0 = closed -> upper.
                    targets[dof] = hi + norm[act_idx] * (lo - hi)
        return targets

    def set_actuator_targets(self, normalized: np.ndarray) -> None:  # pragma: no cover
        targets = self._expand_targets(normalized).astype(np.float32)
        self._gym.set_actor_dof_position_targets(self._env, self._actor, targets)

    def step(self) -> None:  # pragma: no cover - needs GPU runtime
        gym, sim = self._gym, self._sim
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        if self._viewer is not None:
            gym.step_graphics(sim)
            gym.draw_viewer(self._viewer, sim, True)
            gym.sync_frame_time(sim)

    def close(self) -> None:  # pragma: no cover - needs GPU runtime
        if self._gym is None:
            return
        if self._viewer is not None:
            self._gym.destroy_viewer(self._viewer)
        if self._sim is not None:
            self._gym.destroy_sim(self._sim)
        self._gym = self._sim = self._env = self._actor = self._viewer = None
