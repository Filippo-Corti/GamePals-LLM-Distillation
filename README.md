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

# Some data

Teacher labels:
FC:  42	CNO:   6	OW:   0	CW:   2	SW:   0	

Teacher Latency:
5.370 +- 2.718 (Median: 4.818)

Qwen 1.5B (untrained):
Labels: FC:   4	CNO:   6	OW:  11	CW:  14	SW:  15
EM: 176 False (97%) - 6 True (3%)
Edit Distance: 0.547 +- 0.250 (Median: 0.568)
F1: 0.484 +- 0.349 (Median: 0.500)
No interesting patterns
Latency: 0.256 +- 0.201 (Median: 0.195)

Qwen 1.5B (trained):
Labels: FC:  38	CNO:  10	OW:   0	CW:   2	SW:   0	
EM: 134 False (74%) - 48 True (26%)
Edit Distance: 0.291 +- 0.303 (Median: 0.200)
F1: 0.721 +- 0.331 (Median: 0.873)
No interesting patterns
Latency: 0.219 +- 0.127 (Median: 0.186)

Qwen 0.5B (untrained):
Labels: FC:   0	CNO:   0	OW:   0	CW:   1	SW:  49	
EM: 176 False (100%) - 0 True (0%)
Edit Distance: 0.803 +- 0.122 (Median: 0.803)
F1: 0.174 +- 0.206 (Median: 0.0)
No interesting patterns
Latency: 0.425 +- 0.793 (Median: 0.167)

Qwen 0.5B (trained):
Labels: FC:  34	CNO:  11	OW:   0	CW:   4	SW:   1	
EM: 133 False (73%) - 49 True (27%)
Edit Distance: 0.293 +- 0.300 (Median: 0.201)
F1: 0.730 +- 0.315 (Median: 0.809)
No interesting patterns
Latency: 0.164 +- 0.090 (Median: 0.142)

---

# Convert HF to OLLAMA

cd llama.cpp (or just download the built version)

python convert_hf_to_gguf.py ../models/final --outfile qwen-commanding.gguf --outtype f16

cd llama-build
.\llama-quantize.exe ../qwen-commanding.gguf ../qwen-commanding-q4.gguf Q4_K_M

Create a Modelfile (like the one in the repo)

ollama create qwen-commanding -f .\Modelfile2


---

For FunctionGemma, you do not need to convert to GGUF. YOu can just do 

ollama create fgemma-commanding -f Modelfile2 --quantize q4_k_m

> Currently going with no quantization