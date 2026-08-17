# Software environment

The public package supports Python 3.10+. Runtime dependencies are declared once in `pyproject.toml`; `requirements.txt` and `environment.yml` install that same project definition rather than maintaining duplicate dependency lists.

## Reference environment used by the repeated paper runs

The saved runtime metadata from the five-realization experiment records the following representative environment:

- Python 3.10.19
- Linux 6.8 / glibc 2.35
- PyTorch 2.7.1 + CUDA 11.8
- NumPy 2.2.5
- NVIDIA CUDA device (`cuda:0`)

These versions document the paper-run environment; they are not intended as the only supported versions. For strict reproduction, matching the paper-run environment as closely as practical is recommended. Each run directory records the runtime software, device, seed, and experiment configuration metadata needed to interpret the result.

## CPU versus GPU

Small diagnostic runs can run on CPU. The full 85-run reproduction is computationally intensive and is intended for CUDA-capable hardware.
