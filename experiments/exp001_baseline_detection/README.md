# Experiment 001 — Baseline Scene Detection

## Objective

Determine what environmental information can be extracted from ordinary RGB
video using a lightweight pretrained object detector.

## Input

Video recorded from:

- WeightGait
- controlled indoor scenes
- later mobile camera recordings

## Model

YOLO26n pretrained detector.

## Outputs

- annotated video
- detected objects
- confidence values
- processing FPS
- qualitative observations

## Important limitation

Object detection does not directly provide:

- doorway width
- safe walking path
- exact distance
- spatial difficulty
- appropriate therapeutic cue

These require additional spatial reasoning.

## Research question

What environmental information can be reliably extracted from ordinary RGB
video before introducing explicit spatial reasoning?