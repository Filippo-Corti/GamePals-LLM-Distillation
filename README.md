# Knowledge Distillation for GamePals LLM

Pipeline:

1. [X] Data Collection
2. [X] Data Augmentation
3. [X] Dataset Filtering
4. [X] Teacher Action Generation
5. [X] Student Training Setup
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
- Test the trained student again, removing |im_end|
- Try to port the student to ollama and see if results are faster
- Evaluate the student on the 50 instances
- If results are alright, run the student on all instances

Next steps:
- Compare the results from student and teacher
- Do it all again with FunctionGemma

REMEMBER: validity metrics are computed LATER ON

---

# Convert HF to OLLAMA

cd llama.cpp (or just download the built version)

python convert_hf_to_gguf.py ../models/training/final --outfile qwen-commanding.gguf --outtype f16

.\llama-quantize.exe qwen-model.gguf qwen-model-q4.gguf Q4_K_M

Create a Modelfile (like the one in the repo)

ollama create qwen-commanding -f .\Modelfile