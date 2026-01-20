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


Annotation notes:
- The model is not precise with timings.
- Some user commands are not possible for the model (at level 2) - or are with very weird phrasings

Reasonable fixes and new pipeline:
1) Move to Level 3 (from Level 2)
2) Improve the prompt for game actions a little bit (on Sequentiality)
3) Keep the commands as they are: no time to regenerate them
4) Save all the commands already clustered and with a field "selected_for_labeling"
5) Run the Teacher only on selected_for_labeling -> evaluate them using the rubric. Use reasoning 'low', if many empty returns use reasoning 'none' and temperature 0.0.
6) Hope everything is at least decent on step 5 -> do the same for a student

Then we'll think about the training...