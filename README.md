# Knowledge Distillation for GamePals LLM

Pipeline:

1. [X] Data Collection
2. [X] Data Augmentation
3. [ ] Dataset Filtering
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
- Finish up the command-execution template for level 2
- Finish up the tool calls for level 2
- Test the OpenAI API Client with the labeling task
- If everything is ok, start working on the API Client for the Student

---


Some old notes:


Notes on LLM-Distillation code:

OpenAIClient può essere utilizzato:
- Come Batch, per fare UserCommands
- Come Sequential, per fare Labeling del Teacher
Non può essere usato come Student -> Serve un altro Client, tipo HuggingFaceClient che gestisca le chiamate sequenziali dello student
esattamente allo stesso modo:
 - Questo motiva un Client generico, da valutare in seguito

Faccio tutto nel notebook, per il momento (al di fuori dei client)

Nella pratica:
- Passo il dataset al client e il system prompt già parsato per Doom, per poter generare gli user commands
- Salvo gli user commands semplicemente in un array

Dopodiché:
- Costruisco un nuovo dataset che unisce:
 - User Command e il suo Game State
- Scrivo di nuovo il prompt
- Costurisco un nuovo client a cui passo il nuovo dataset. Risultato sono le label (formattate in una certa maniera, come lista di Action)
 - Usando le Tool Call!!!
- Ottengo un array di labels, ognuna associata sia a id dello user command che al game state 

---

Poi mi devo occupare dello student…
 Devo capire come si fa il fine tuning 
  E' idealmente una funzionalità del client HuggingFaceClient





