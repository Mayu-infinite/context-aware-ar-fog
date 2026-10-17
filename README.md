# Context-Aware AR for Freezing of Gait

A research project exploring whether real-time visual-spatial understanding
of the environment can be used to select appropriate external walking cues
for situations associated with Freezing of Gait (FoG).

## Core pipeline

Camera
↓
Scene Perception
↓
Spatial State
↓
Context Understanding
↓
Action Policy
↓
Visual / Audio Cue

## Initial development platform

- EndeavourOS Linux
- Python
- uv
- NVIDIA GPU
- OpenCV
- Ultralytics YOLO
- PyTorch
- Unity
- Meta Quest 3

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