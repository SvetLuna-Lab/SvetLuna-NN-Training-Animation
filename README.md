# SvetLuna — NN Training Animation

**Neural Network Training Visualization — an educational Python/Matplotlib demo illustrating decreasing loss, increasing accuracy, and a schematic network representation.**

**Author:** Svetlana Romanova — SvetLuna Studio  
**LinkedIn:** https://www.linkedin.com/in/svetlana-romanova-418596387

---

## Overview
This project provides a concise visual demonstration of neural network training dynamics. It shows a loss curve decreasing, an accuracy curve increasing, and a schematic network whose activations evolve as training progresses. The goal is to give beginners and educators an intuitive, visual explanation of how models learn.


## 🔧 Python demo code

In addition to the video and archive, this repository now contains a small
pure-Python / Matplotlib demo:

- `src/nn_training_animation.py` — generates synthetic loss/accuracy curves and
  an animated schematic neural network.
- `nn_training_animation.mp4` — an exported animation (example output).


### Quickstart

Create a virtual environment (optional) and install dependencies:

```bash
pip install -r requirements.txt

Run the animation interactively:

python -m src.nn_training_animation

Or render and save an MP4 (requires ffmpeg to be installed):

python -m src.nn_training_animation save

The legacy project archive is kept as:

SvetLuna-NN-Animation-Project-ready.zip — original project bundle.


## Contents
- `src/generate_animation.py` — script to generate the animation (MP4).  
- `example/` — example output (store large video files as Release assets; do not keep large binaries in Git history).  
- `.github/workflows/generate.yml` — optional GitHub Actions workflow for automated generation.  
- `requirements.txt` — required Python packages.  
- `docs/` — design notes and visualization rationale.  
- `assets/` — preview images and thumbnails.  
- `CONTRIBUTING.md`, `LICENSE`, `.gitignore`

## Quick start (local)
1. Clone repository:
```bash
git clone https://github.com/<your-username>/SvetLuna-NN-Animation-Project.git
cd SvetLuna-NN-Animation-Project
Create and activate a virtual environment:

bash
Копировать код
python -m venv venv
# Unix / macOS
source venv/bin/activate
# Windows (PowerShell)
# .\venv\Scripts\Activate.ps1
Install dependencies:

bash
Копировать код
pip install -r requirements.txt
Generate the animation:

bash
Копировать код
python src/generate_animation.py --frames 35 --fps 1 --out example/nn_training_animation.mp4
Recommended project structure
css
Копировать код
SvetLuna-NN-Animation-Project/
├── .github/
│   └── workflows/
│       └── generate.yml
├── src/
│   └── generate_animation.py
├── example/
│   └── nn_training_animation.mp4  # keep out of Git history; use Release or artifact
├── docs/
├── assets/
├── tests/
├── requirements.txt
├── README.md
└── LICENSE
CI / Automation
A GitHub Actions workflow (.github/workflows/generate.yml) can:

install Python dependencies,

run the generator script,

upload the produced MP4 as an action artifact (or attach to a Release).

Note: action runs will produce the video artifact without adding large binaries to the repo history.

Video storage recommendations
Do not commit large MP4 files into the repository history. Use one of:

GitHub Releases (attach MP4 as a release asset),

Git LFS (if you must keep large files in the repo),

store video in external storage and provide a download link in README.md.

For reproducibility, keep the generator script (src/generate_animation.py) and dependencies in the repo.

Contributing
Fork the repository.

Create a branch: feature/<short-description>.

Make changes and add tests if applicable.

Open a pull request describing the changes.

Maintain clear commit messages and respect the code style used in src/.

Suggested issue labels: bug, enhancement, documentation, good first issue, help wanted.

License
MIT — see LICENSE.

Contact
Svetlana Romanova — SvetLuna Studio
LinkedIn: https://www.linkedin.com/in/svetlana-romanova-ai

Note: This repository is an educational demonstration. The visual style is intentionally simple to highlight concepts rather than production-ready visuals.

Poetic closing: where curves converge and lights stabilise, the model learns — quietly, reliably, and with purpose.

