## Input:
AIMED_AT:
    type: Wall
    distance: 100.33
    interactable: no

MONSTERS (count=1):
    - (MONSTER_1, DoomImp, 60, 968.24, -10.56, 12.32)

INVENTORY:
    current_slot: 2
    weapons:
    - (1, Fist, 0)
    - (2, Pistol, 33)

## Output:
{"command": "Shoot the imp in front of me", "intent": "Eliminate the nearby hostile monster", "explicitness": 0.75, "atomicity": 0.45, "contextuality": 0.85}
{"command": "Aim at the monster", "intent": "Align the crosshair with the visible monster", "explicitness": 0.8, "atomicity": 0.3, "contextuality": 0.85}
{"command": "Help me", "intent": "Receive assistance in handling the current combat threat", "explicitness": 0.05, "atomicity": 0.55, "contextuality": 0.95}