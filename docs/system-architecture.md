# System Architecture

The main pipeline is the **label-free spatial cue engine**. It never names objects. It
models what the walkable space ahead demands of the patient and cues only when a
Parkinsonian value function predicts a slowdown or freeze.

```
Quest depth (or RGB-D / simulation) + head pose
        │
        ▼
2.5D traversability grid        log-odds, world-anchored, 10 cm cells, rolling 10 x 10 m
(floor / obstacle / drop-off)   floor height from a height histogram (no plane labels)
        │
        ▼
Training-free critic            V_HC = gamma^D, D = cost-weighted geodesic distance to the
(value heatmap)                 frontier ahead (multi-goal Dijkstra, exact, no learning)
        │                       V_PD = V_HC * exp(-kappa * r), kappa ∝ FOG factor
        ▼                       r = max(turn ratio, narrowness ratio)  -> "value canyon"
Predicted route + GEN model     Go/No-go speed from dV along the route (Bhattacharya et al.
(risk of slowing / freezing)    2026, Eqs. 9-10, 28-32); risk = 1 - v_PD / v_HC
        │
        ▼
Cue controller                  hysteresis, initiation mode when standing still,
                                footsteps world-anchored on the floor (LIP width, Eq. 26),
                                metronome at the LIP step rhythm (Eq. 25)
```

## Where the code lives

| Layer | Python reference | Quest 3 (Unity) |
| --- | --- | --- |
| Depth → grid counts | `spatial/geometry.py`, `spatial/occupancy.py` | `Shaders/DepthToGrid.compute` |
| Grid, critic, route, GEN, cues | `spatial/value.py`, `path.py`, `gen.py`, `cues.py`, `engine.py` | `Runtime/Core/*.cs` (pure C#) |
| Rendering | `spatial/viz.py` (researcher video) | `FootstepCueRenderer`, `MetronomeCue`, `ValueHeatmapView` |
| Personalisation | `configs/spatial_engine.yaml` | `CafogEngineBehaviour.configJson` |

The C# core is a line-by-line port. `scripts/export_parity_trace.py` and
`unity/CafogQuest/Tools/CoreParity` replay identical inputs through both and require
identical cue phases, footstep counts, risk values and footstep positions.

## Why no Gaussian splatting in the loop

Gaussian splatting produces photorealistic novel views, but it needs seconds to minutes
of optimisation per scene on a desktop GPU and its output (view-dependent blobs) is not
what a walking-safety decision needs. The cue depends on free space, clearance, turns and
drop-offs, so the headset fuses depth into a metric grid in under a millisecond on the
GPU. For a full 3D model of a recorded space, `spatial/voxels.py` fuses coloured depth
into a voxel map and exports a PLY (`scripts/run_scannet.py`); that model can be fed to
an offline splatting tool for visualisation if needed.

## Latency (measured on the development laptop CPU, 90° corridor walk, median)

| Stage | Python reference | Quest path |
| --- | --- | --- |
| Back-projection | 0.75 ms | GPU compute shader |
| Grid update | 0.46 ms | C# core |
| Value field (EDT + Dijkstra, 100 x 100 cells) | 5.3 ms | C# core |
| Route + GEN risk | 0.68 ms | C# core |
| Cue controller | 0.03 ms | C# core |
| Total | 7.3 ms (p95 8.5 ms) | C# core total 1.2-3.5 ms mean across scenes |

The C# core runs on a background thread on the headset, so the Quest 3 CPU being slower
than a laptop does not affect the render loop; it only bounds the cue update rate.

The cue fires about 2.4 m before a turn and 2.9 m before a narrow door, i.e. about 3 s of
walking ahead, so a GPU readback delay of 1-2 frames is irrelevant to the patient.

## Phone-camera prototype

A phone has neither a depth sensor nor head tracking, so `sources/monocular.py` supplies
both before the same engine runs:

- metric depth from a pretrained network (Depth Anything V2 Metric-Indoor Small, used as is),
- gravity from the floor plane fitted in every frame (no drift); the plane also fixes the
  network's depth scale (true camera height / fitted height, smoothed), which on the test
  corridors was ~0.5 — the network alone over-estimates distance about twofold,
- heading and walking motion from frame-to-frame ORB + PnP odometry on the metric depth,
  re-levelled on the floor every frame.

This is a verification tool: monocular depth is less accurate than the Quest's depth
sensor, and the depth network dominates latency on a laptop CPU.

## Earlier pipeline

The first prototype (YOLO object detection + context rules) was removed after commit
`1e9c25a`; it labelled objects, which patients should not see, and object classes do not
determine walking demand. `git checkout 1e9c25a -- src/cafog` restores it for comparison.
