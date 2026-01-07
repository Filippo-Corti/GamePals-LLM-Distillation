## Input:
AIMED_AT:
    type: Wall
    distance: 330.86
    interactable: yes

MONSTERS (count=0):

INVENTORY:
    current_slot: 1
    weapons:
    - (1, Fist, 0)
    - (2, Pistol, 50)

## Output:
{"command": "Go to the button and activate it", "intent": "Move closer to the interactable wall and trigger its interaction", "explicitness": 0.8, "atomicity": 0.4, "contextuality": 0.85}
{"command": "Switch to my best weapon", "intent": "Improve combat readiness by changing weapon", "explicitness": 0.3, "atomicity": 0.8, "contextuality": 0.1}