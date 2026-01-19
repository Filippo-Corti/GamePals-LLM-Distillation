# Knowledge Distillation for GamePals LLM

Pipeline:

1. [X] Data Collection
2. [X] Data Augmentation
3. [X] Dataset Filtering
4. [ ] Teacher Action Generation
5. [ ] Student Training Setup
6. [ ] Teacher vs Student Comparison
7. [ ] Repeat with Variations


---

Game-Specific Tasks:

1) Collect Game States
2) Mapping from a Game State to the Feature Vector used for filtering 
3) Functions that make Perturbations of the Game State
4) Some parts of the Data Augmentation prompt
5) Some parts of the Main Task prompt


---

Short-term TODOs:
- Run a small instance of inputs using the Teacher IN SEQUENTIAL MODE.
- Test the performance of the Teacher (REMEMBER TO STORE AND CHECK THE LATENCY).
  - If feeling like we can do better with better prompt, edit prompt and restart
- Assuming satisfied enough, keep the labels.
- Run all inputs using the teacher IN BATCH MODE. I should re-evaluate but I could also just replace the outputs together with the evaluation.
- Now we have a dataset of reference outputs (SAVE THEM).

Next steps:
- Run a student with no training (zero-shot) and evaluate how much worse it is.
- Train a student. Run again.
EVERY TIME: USE THE SAME SELECTED INPUTS.


---
