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
```tool-call
execute({
  "actions": [
    {
      "action": "direction_x",
      "value": -1.0,
      "duration": 0.12,
      "blocking": false
    },
    {
      "action": "direction_y",
      "value": 1.0,
      "duration": 0.14,
      "blocking": true
    },
    {
      "action": "fire",
      "value": 1.0,
      "duration": 0.5,
      "blocking": true
    }
  ]
})
```