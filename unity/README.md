# Cafog for Meta Quest 3

Label-free, value-based Freezing-of-Gait cueing in passthrough mixed reality. The headset
maps walkable space from the Depth API, predicts where a Parkinsonian walker would slow
down or freeze (value canyon + GEN model), and places footprints on the real floor ahead.

## Contents (`CafogQuest/Assets/Cafog`)

| Path | Purpose |
| --- | --- |
| `Runtime/Core/` | Pure C# engine (no UnityEngine; `Cafog.Core.asmdef`). Bit-for-bit port of `src/cafog/spatial`. |
| `Runtime/Quest/QuestDepthGridSource.cs` | Dispatches `DepthToGrid.compute` on the Depth API frame, async readback of grid counts. |
| `Runtime/Quest/CafogEngineBehaviour.cs` | Runs the engine on a background thread, publishes the latest cue. |
| `Runtime/Quest/FootstepCueRenderer.cs` | World-anchored footprints or transverse bars on the floor. |
| `Runtime/Quest/MetronomeCue.cs` | DSP-clock metronome at the LIP step rhythm. |
| `Runtime/Quest/ValueHeatmapView.cs` | Researcher view of V_PD / V_HC on the floor (keep off for patients). |
| `Shaders/DepthToGrid.compute` | Depth → world points → floor estimate → per-cell obstacle/floor/drop-off counts. |
| `Shaders/CafogFootprintUnlit.shader` | Soft-edged transparent cue material (single-pass instanced). |
| `Config/cafog_config.json` | Patient profile exported from `configs/spatial_engine.yaml`. |

## Project setup (Windows, Unity 6 or 2022.3 LTS)

1. Create a Unity project with the **Universal 3D** template at `unity/CafogQuest`
   (keep the existing `Assets/Cafog` folder).
2. Install **Meta XR Core SDK** (v71 or newer) and **Meta MR Utility Kit** from the Asset
   Store / Package Manager. Run *Meta > Tools > Project Setup Tool* and apply all fixes.
3. Build settings: Android, ARM64, IL2CPP, Vulkan, OpenXR with the Meta Quest feature group.
4. Scene:
   - Add `OVRCameraRig` (or the *Camera Rig* building block). Set **Tracking Origin Type =
     Floor Level**. Enable **Passthrough** (OVRManager → Insight Passthrough, plus an
     `OVRPassthroughLayer` set to Underlay) and clear the camera to transparent black.
   - Add `EnvironmentDepthManager` (Meta XR Core SDK). Occlusion can stay off; Cafog only
     needs the depth texture it publishes.
   - Create an empty `Cafog` object with:
     - `QuestDepthGridSource` → *Depth To Grid* = `Shaders/DepthToGrid.compute`
     - `CafogEngineBehaviour` → *Depth Source* = the component above, *Head* =
       `CenterEyeAnchor`, *Config Json* = `Config/cafog_config.json`
     - `FootstepCueRenderer` → *Engine* = the behaviour, *Material* = a material using
       `Cafog/FootprintUnlit`
     - `MetronomeCue` → *Engine* = the behaviour
   - Optional for researchers: a child object with `ValueHeatmapView` and a material using
     `Unlit/Transparent`.
5. Permissions: the Depth API needs **Scene** permission (`com.oculus.permission.USE_SCENE`).
   The Project Setup Tool adds it; grant it on first launch.
6. Build and run on Quest 3.

## Personalising for a patient

Edit `configs/spatial_engine.yaml` (profile preset, step length, metronome speed, trigger
distance, comfortable passage width), then

```bash
uv run python scripts/export_unity_config.py configs/spatial_engine.yaml
```

and rebuild. At runtime a companion app can push the same JSON through
`CafogEngineBehaviour.ApplyConfigJson(json)`.

## Verifying on device

1. Enable `ValueHeatmapView` and walk a corridor: the coloured map must line up with the
   real floor and walls. If it appears mirrored front-to-back, toggle *Flip Y* on
   `QuestDepthGridSource` (depth texture origin differs between graphics APIs).
2. Walk towards a 90° corner at normal pace with the `pd_fog` preset: footprints should
   appear about 2-3 m before the corner and continue around it; the metronome starts with
   them and stops about a second after the turn is completed.
3. Stand still facing a doorway: an initiation cue (first steps + metronome) appears when
   the door is within 1.5 m.

## Checking the C# core without Unity

```bash
uv run python scripts/export_parity_trace.py
dotnet run -c Release --project unity/CafogQuest/Tools/CoreParity -- experiments/exp007_csharp_parity/results
```

Prints `PARITY OK` when the C# engine reproduces the Python engine frame by frame.
