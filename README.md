# SvetLuna — NN Training Animation

Neural Network Training Visualization — an educational Python/Matplotlib demo
illustrating decreasing loss, increasing accuracy, and a schematic network
representation.

**Author:** Svetlana Romanova — SvetLuna Studio  
**LinkedIn:** https://www.linkedin.com/in/svetlana-romanova-418596387

---

## 🌟 Overview

This project provides a concise visual demonstration of neural network training
dynamics. It shows:

- a **loss** curve that decreases over epochs,  
- an **accuracy** curve that increases,  
- a simple **network schematic** whose connections “light up” as training
  progresses.

The goal is to give beginners and educators an intuitive, visual explanation of
how models learn.

---

## 🔧 Python demo code

The repository contains a small pure-Python / Matplotlib demo:

- `src/nn_training_animation.py` — generates synthetic loss/accuracy curves
  and an animated schematic neural network;
- `nn_training_animation.mp4` — an exported animation (example output);
- `SvetLuna-NN-Animation-Project-ready.zip` — original project bundle.

---

## 🚀 Quickstart

Create and (optionally) activate a virtual environment, then install
dependencies:

```bash
pip install -r requirements.txt
Run the animation interactively

python -m src.nn_training_animation
Or render and save an MP4 (requires ffmpeg to be installed):

python -m src.nn_training_animation save
If saving fails (no ffmpeg), you can still watch the animation in a window.

📄 License & use
This repository is intended for education and self-study.
If you use the animation in your teaching materials or talks, a short mention
of the author (Svetlana Romanova — SvetLuna Studio) is welcome.



