# Project Plan

## Phase 1 — Spatial perception (done)

Depth + head pose
↓
2.5D traversability grid (floor, obstacles, drop-offs; no object labels)

## Phase 2 — Value model (done)

Traversability grid
↓
training-free critic (value heatmap) + Parkinsonian value canyon
↓
Go/No-go prediction of slowing or freezing along the route ahead

## Phase 3 — Cue (done)

Predicted freeze risk
↓
footsteps on the floor (+ metronome), initiation cue when standing still

## Phase 4 — Simulation and datasets (done)

Paper turning angles and doorways in simulation; ScanNet RGB-D replay

## Phase 5 — Mobile prototype (in progress)

Phone camera
↓
monocular metric depth + floor-anchored visual odometry on the laptop
↓
same engine and cue

## Phase 6 — Quest 3

Quest Depth API
↓
GPU grid binning + C# engine (parity-tested against Python)
↓
footsteps in passthrough

## Phase 7 — Evaluation

Healthy participants first; clinical evaluation only with appropriate
supervision and approvals

## Phase 8 — Optional gait integration

Environment
+
gait state
↓
context-aware FoG cueing
