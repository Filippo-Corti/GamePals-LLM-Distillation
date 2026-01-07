## Input:
Game State:
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

User Command: 
Go to the button and activate it

## Output:
```tool-call
execute({
  "actions": [
    {
      "action": "movement_y",
      "value": 1.0,
      "duration": 1.25,
      "blocking": true
    },
    {
      "action": "interact",
      "value": 1.0,
      "duration": 0.2,
      "blocking": true
    }
  ]
})
```