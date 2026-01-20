Distances are expressed in game units.   
Angles are expressed in degrees.

```
AIMED_AT:
    type: <entity_type>
    distance: <float>
    interactable: <yes/no>

MONSTERS:  
    - (<target_id> <monster_type>, <health>, <distance>, <relative_angle>, <relative_pitch>)
    
INVENTORY:
    current_slot: <int>
    weapons:    
    - (<slot>, <weapon_name>, <ammo_count>)
```