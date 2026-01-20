## Input:
Game State:
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

User Command: 
Help me

## Output:
ROTATE_TO_TARGET MONSTER_1
FIRE 1.0
