# Knowledge Distillation for GamePals LLM

Partial automation is an interaction paradigm in which users delegate part of
a system’s control to a software agent. 

Prior work has shown its potential to
improve accessibility in video games by assisting players with disabilities in real-
time control. However, experimental evidence also indicates that the lack of
communication between user and software agent can lead to confusion and mis-
understandings. 

To address this limitation, large language models (LLMs) can
interpret natural language voice commands and translate them into executable
game actions. Although frontier LLMs are capable of this task, real-time game
control imposes strict low-latency requirements, motivating the use of local mod-
els. 

In this work, we formalize human-commanded game control and distill the
knowledge of a frontier teacher model (GPT-5.1) into two smaller local student
models (Qwen2.5-1.5B-Instruct and Qwen2.5-0.5B-Instruct), evaluating whether
they preserve the teacher’s capabilities while reducing inference latency. 
Preliminary results show over a 20x reduction in response time while maintaining strong
performance on more than 90% of the test set. 

---

# Knowledge Distillation Pipeline

The full KD pipeline proposed for the task of **human-commanded game control** is contained in the 8 notebooks in this repository:

1. [`Game State Processing`](01_game_state_processing.ipynb): build a dataset of suitable game states.
2. [`User Commands Generation`](02_user_command_generation.ipynb): generate user commands associated with the game states, using the teacher model.
3. [`Dataset Preparation`](03_dataset_preparation.ipynb): build the dataset of <game state, user command> pairs.
4. [`Dataset Labeling`](04_dataset_labeling.ipynb): run inference on the teacher model, to get labels for each input pair.
5. [`Untrained Student`](05_untrained_student.ipynb): run inference on the untrained student model, as baseline.
6. [`Student Training`](06_student_training.ipynb): train the student on the human-commanded game control task.
7. [`Trained Student`](07_trained_student.ipynb): re-run inference on the student model, after training.
8. [`Model Comparisons`](08_model_comparisons.ipynb): compare results between teacher and student, as well as between different students.

A representation of the KD Pipeline by Gemini:

![KD Pipeline](./data/img/kd-pipeline.png)

---

# Content of this Repository

This repository has the following structure:

- `01..08_notebooks` contain the KD pipeline, as presented above.
- [`data`](data) contains all data used and generated throughout the experiments.
- [`prompts`](prompts) contains the generic templates for the prompts and filling parts for *The Ultimate Doom*.
- [`src`](src) contains utility classes and functions used throughout the notebooks.


---

Author: Filippo Corti

Natural Language Procesing Project - A.Y. 2025/2026

#### Useful References:

- [The GamePals Project](https://everywarelab.di.unimi.it/index.php/15-research-projects/233-games-accessibility)
- [Interviews to People using Shared Control](https://arxiv.org/abs/2601.11218)
- [Partial Automation vs Human Cooperation - a study](https://arxiv.org/pdf/2509.02132)

