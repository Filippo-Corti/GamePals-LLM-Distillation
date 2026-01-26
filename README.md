# Knowledge Distillation for GamePals LLM

Pipeline:

1. [X] Data Collection
2. [X] Data Augmentation
3. [X] Dataset Filtering
4. [X] Teacher Action Generation
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
- Re-run the student on the small dataset. I need to use Ollama, not HuggingFace (it should be quicker - but equivalent for everything else)
- Try (at home) the training, on the small dataset.
- Run the trained student. If possible, use Ollama.

Next steps:
- Run all inputs using the teacher IN BATCH MODE. I should re-evaluate but I could also just replace the outputs together with the evaluation.


REMEMBER: validity metrics are computed LATER ON

---