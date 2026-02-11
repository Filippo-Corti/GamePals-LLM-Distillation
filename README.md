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
- Run a comparison notebook for differences between teacher and Qwen
- Run all tests over again with FunctionGemma

---

# Convert HF to OLLAMA

cd llama.cpp (or just download the built version)

python convert_hf_to_gguf.py ../models/final --outfile qwen-commanding.gguf --outtype f16

cd llama-build
.\llama-quantize.exe ../qwen-commanding.gguf ../qwen-commanding-q4.gguf Q4_K_M

Create a Modelfile (like the one in the repo)

ollama create qwen-commanding -f .\Modelfile


---

For FunctionGemma, you do not need to convert to GGUF. YOu can just do 

ollama create fgemma-commanding -f Modelfile2 --quantize q4_k_m

> Currently going with no quantization