### Forward/Backwards Movement ('movement_y'):
movement_y represents the action of moving the avatar forward or backwards.
#### How to use:
- set `action='movement_y'`
- set `value` in range (0.0, 1.0] to move forward
- set `value` in range [-1.0, 0.0) to move backward
- after `duration` seconds, movement_y will be automatically reset to 0.0
#### When to use:
- When the player asks for movement towards a target
- When the player asks for movement away from a target
- When the player asks for specific movement traces (for example, 'move forward for a couple of seconds')
#### Extra information:
- Base vertical movement speed when `value=1.0` is 290 units/s.


### Lateral Movement ('movement_x'):
movement_x represents the action of moving the avatar laterally, left or right.
#### How to use:
- set `action='movement_x'`
- set `value` in range (0.0, 1.0] to move laterally to the right
- set `value` in range [-1.0, 0.0) to move laterally to the left
- after `duration` seconds, movement_x will be automatically reset to 0.0
#### When to use:
- When the player asks for movement towards a target
- When the player asks for movement away from a target
- When the player asks for specific movement traces (for example, 'move left for a couple of seconds')
#### Extra information:
- Base lateral movement speed when `value=1.0` is 290 units/s.


### Vertical Orientation ('direction_y'):
direction_y represents the action of orientating (rotating) the crosshair vertically.
#### How to use:
- set `action='direction_y'`
- set `value` in range (0.0, 1.0] to raise the crosshair
- set `value` in range [-1.0, 0.0) to lower the crosshair
- after `duration` seconds, direction_y will be automatically reset to 0.0
#### When to use:
- When the player asks to aim towards a target that stands on a different vertical height
- When the player asks for specific direction traces (for example, 'look up').
#### Extra information:
- Base vertical direction speed when `value=1.0` is 60 degrees/s.


### Horizontal Orientation ('direction_x'):
direction_x represents the action of orientating (rotating) the crosshair horizontally.
#### How to use:
- set `action='direction_x'`
- set `value` in range (0.0, 1.0] to rotate towards the right
- set `value` in range [-1.0, 0.0) to rotate towards the left
- after `duration` seconds, direction_x will be automatically reset to 0.0
#### When to use:
- When the player asks to aim towards a target that stands to their left or right.
- When the player asks for specific direction traces (for example, 'turn around').
#### Extra information:
- Base horizontal direction speed when `value=1.0` is 60 degrees/s.


### Run ('run'):
When active, run increases the speed of movement by doubling it. 
#### How to Use:
- set `action='run'`
- set `value=1.0` to start running
- after `duration` seconds, run will be automatically reset to 0.0
- use `blocking` appropriately to make sure that movement happens while run is active
#### When to use:
- When the player asks for it explicitly.
- When you want to move faster due to urgency.


### Fire ('fire'):
fire represents the action of shooting the currently selected weapon. 
#### How to Use:
- set `action='fire'`
- set `value=1.0` to start shooting
- after `duration` seconds, fire will be automatically reset to 0.0
#### When to use:
- When the player asks for it, explicitly or implicitly.


### Jump ('jump'):
jump represents the action of jumping. 
#### How to Use:
- set `action='jump'`
- set `value=1.0` to express the intent to jump
- set `duration=0.2s` to make sure the button is pressed long enough to execute the jump
#### When to use:
- When the player asks for it, explicitly or implicitly.


### Interact/Use ('interact'):
interact represents the action of interacting with an interactable element of the environment (e.g., a door or a button). 
#### How to Use:
- set `action='interact'`
- set `value=1.0` to express the intent to interact
- set `duration=0.2s` to make sure the button is pressed long enough to execute the interaction
#### When to use:
- When the player asks for it, explicitly or implicitly.


### Switch to Previous Weapon ('prev_weapon')
prev_weapon represents the action of switching to the first available weapon to the left of the currently assigned one.
If the currently assigned one is the first weapon, it will swap to the last one.
Only weapons with at least 1 available ammunition are considered by the game for the swap.
#### How to Use:
- set `action='prev_weapon'`
- set `value=1.0` to express the intent to swap to the previous weapon
- set `duration=0.2s` to make sure the button is pressed long enough to execute the swap
#### When to use:
- When the player asks for it, explicitly or implicitly.


### Switch to Next Weapon ('next_weapon')
next_weapon represents the action of switching to the first available weapon to the right of the currently assigned one.
If the currently assigned one is the last weapon, it will swap to the first one.
Only weapons with at least 1 available ammunition are considered by the game for the swap.
#### How to Use:
- set `action='next_weapon'`
- set `value=1.0` to express the intent to swap to the next weapon
- set `duration=0.2s` to make sure the button is pressed long enough to execute the swap
#### When to use:
- When the player asks for it, explicitly or implicitly.

