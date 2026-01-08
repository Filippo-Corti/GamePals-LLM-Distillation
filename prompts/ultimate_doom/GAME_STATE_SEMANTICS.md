The game state for The Ultimate Doom contains the following information about the environment surrounding the player:

- `AIMED_AT` represents what is currently behind the player's crosshair. This can either be a monster, another actor, a wall, the floor, the ceiling or unknown.  
  The field `interactable` determines whether the action 'interact' has an effect when execute next to the object. For instance, if the crosshair is on a button `interactable` will likely be true.
- `MONSTERS` is the list of monsters surrounding the player. For each monster, the game state reports information about their type and their location (both in the 3D space and w.r.t the player's crosshair).  
  In particular, the fields `relativeAngle` and `relativePitch` determine the on-screen distance between the crosshair and the monster:
  - `relativeAngle` < 0 implies that the monster is to the left of the crosshair; `relativeAngle` > 0 implies that the monster is to the right of the crosshair.
  - `relativePitch` < 0 implies that the monster is below the crosshair; `relativePitch` > 0 implies that the monster is above the crosshair.
- `INVENTORY` contains the list of available weapons and the currently selected weapon slot. Only weapons with a number of ammunition greater than zero can be selected for use.
