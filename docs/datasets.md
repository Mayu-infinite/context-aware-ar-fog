
---

# 41. `docs/datasets.md`

```markdown
# Datasets

## Initial dataset

WeightGait is the initial computer-vision testbed.

It contains RGB/depth video and simulated gait abnormalities including
freezing and obstacle conditions.

It should not be treated as a clinical Parkinson's disease dataset.

## Purpose in this project

WeightGait is used to test:

- video processing
- scene perception
- temporal processing
- environmental context recognition

## Custom dataset

A project-specific dataset will eventually be recorded.

Each clip may contain:

- scene
- doorway width
- approximate distance
- obstacle
- path configuration
- context
- recommended action
- visual cue
- audio cue
- difficulty

Example:

```json
{
  "scene": "narrow_doorway",
  "door_width": 0.65,
  "door_distance": 1.2,
  "obstacle": false,
  "context": "NARROW_DOORWAY",
  "recommended_action": "CENTER_PATH",
  "visual_cue": "FOOTSTEPS",
  "audio_cue": "RHYTHMIC"
}