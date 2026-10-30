# Context-Aware AR for Freezing of Gait

A research project exploring whether real-time visual-spatial understanding
of the environment can be used to select appropriate external walking cues
for situations associated with Freezing of Gait (FoG).

## Core pipeline: label-free spatial cue engine

No object labels are shown or needed. The engine models what the walkable space ahead
demands of the patient and cues when a Parkinsonian value function predicts a freeze
(Bhattacharya et al. 2026, "Impaired value estimation over space explains turn-induced
freezing of gait").

```
Depth + head pose (Quest 3 Depth API / RGB-D dataset / simulation)
↓
2.5D traversability grid (floor, obstacles, drop-offs)
↓
Training-free critic: value heatmap V_HC = gamma^distance, PD value canyon V_PD
↓
Predicted route + Go/No-go model → risk of slowing / freezing ahead
↓
Footsteps on the floor (+ metronome) before turns, doorways, narrow passages,
and when starting to walk
```

Details: [docs/system-architecture.md](docs/system-architecture.md).
Quest 3 app: [unity/README.md](unity/README.md).

## Quick start

```bash
uv sync
uv run pytest
# Paper turning angles (0/60/90/120/180 deg) + doorways for healthy / PD / PD-FOG profiles
uv run python scripts/run_simulation.py --videos
# Phone camera (IP Webcam app), webcam index, or a recorded walking video.
# --camera-height (m above the floor) fixes the monocular depth scale via the floor plane.
uv run python scripts/run_phone_camera.py http://PHONE_IP:8080/video --camera-height 1.35
uv run python scripts/run_phone_camera.py data/raw/videos/video02.mp4 --every 3
# Replay a ScanNet RGB-D scene (download separately under the ScanNet terms of use)
uv run python scripts/run_scannet.py data/raw/scannet/scene0000_00
# Patient profile -> Quest app
uv run python scripts/export_unity_config.py configs/spatial_engine.yaml
```

`run_simulation.py` writes to `experiments/exp005_spatial_value/results/`:
`simulation_summary.csv`, per-scene videos, `value_maps_hc_vs_pdfog.png` and
`value_canyon_along_route.png` (value and predicted speed along the route; reproduces the
paper's value canyon and freezing at turns for the PD-FOG profile).

`run_scannet.py` writes a researcher video, a per-frame CSV log and a coloured 3D model
(`model.ply`) of the scanned space.

## Development platform

- EndeavourOS Linux (development), Windows (Unity build and deployment)
- Python, uv, NumPy, SciPy, OpenCV, PyTorch + Transformers (phone-camera depth only)
- Unity + Meta XR Core SDK, Meta Quest 3

## Development strategy

1. Offline video analysis
2. Scene perception
3. Spatial context recognition
4. Rule-based action selection
5. Cue generation
6. Real-time phone camera prototype
7. Quest 3 passthrough integration
8. Temporal context
9. Optional gait/FoG integration
10. Healthy participant evaluation
11. Clinical evaluation only with appropriate supervision and approvals

## Important design principle

The system should not merely identify objects.

It should determine:

"What should the user do next given the current spatial context?"
