# Experiment 002 — Rule-Based Context

## Objective

Convert structured environmental information into a high-level walking
context.

The purpose of this experiment is to establish a transparent and
interpretable baseline before introducing a learned context model.

## Input

The input is a `SceneState` containing environmental information such as:

- doorway presence
- doorway distance
- doorway width
- obstacle presence
- obstacle distance
- path width
- turn angle
- walkable path

## Pipeline

```text
Scene Perception
       ↓
   SceneState
       ↓
 ContextEngine
       ↓
 Environmental Context