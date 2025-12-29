# Knowledge Distillation for GamePals LLM

Pipeline:

- [ ] 1. Data Collection
- [ ] 2. Data Augmentation
- [ ] 3. Dataset Filtering
- [ ] 4. Teacher Action Generation
- [ ] 5. Student Training Setup
- [ ] 6. Teacher vs Student Comparison
- [ ] 7. Repeat with Variations


---

Game-Specific Tasks:

1) Collect Game States
2) Mapping from a Game State to the Feature Vector used for filtering 
3) Functions that make Perturbations of the Game State
4) Some parts of the Data Augmentation prompt
5) Some parts of the Main Task prompt


---


Teacher:
- Knowledge Elicitation: 
  - Data Curation (create Inputs)
  - Labeling (create Outputs)

Student:
- Supervised Fine-Tuning
- Distillation:


labelled_dataset = teacher.elicit_knowledge(
    game states dataset,
    transformations (
        filter with clustering,
        create perturbations,
        data curation (add user commands),
        labeling (add game actions)
    )
)





