========================
ACTION DSL GRAMMAR
========================

PROGRAM := LINE+
LINE := [ASYNC] OPERATION
ASYNC := "ASYNC"

------------------------
BASE OPERATIONS
------------------------

SELECT <WEAPON_NAME>                            ; switch to an available weapon
FIRE <SECONDS>                                  ; fire the current weapon
ROTATE <HORIZONTAL_DEGREES> <VERTICAL_DEGREES>  ; rotate the player's view
MOVE <HORIZONTAL_UNITS> <VERTICAL_UNITS>        ; move the player's avatar
JUMP                                            ; jump
INTERACT                                        ; interact with doors, buttons, ...
RUN <ON|OFF>                                    ; activate/deactivate the run, for faster MOVE operations
FAIL <REASON>                                   ; fail to fulfill the command


RULES FOR BASE OPERATIONS:
- FIRE <SECONDS> uses a real-valued duration
- ROTATE angles are in degrees
- MOVE units are relative displacements
- RUN ON and RUN OFF only make sense if there is a MOVE between them
- FAIL must be the ONLY instruction in the program when used

------------------------
COMPLEX OPERATIONS
------------------------

ROTATE_TO_TARGET <TARGET_ID>                    ; rotate view until crosshair is on target
MOVE_TO_TARGET <TARGET_ID>                      ; move until right in front of target
SPRINT <HORIZONTAL_UNITS> <VERTICAL_UNITS>      ; equivalent to RUN ON; MOVE ...; RUN OFF
FIRE_SHOTS <N_SHOTS>                            ; fire a number of shots
LONG_JUMP                                       ; jump a large distance

RULES FOR COMPLEX OPERATIONS:
- Complex operations are atomic shortcuts
- Do NOT expand complex operations into base operations
- Use complex operations when they better match the intent

------------------------
ASYNC VS BLOCKING
------------------------

- All operations are BLOCKING by default
- Prefixing an operation with ASYNC makes it NON-BLOCKING
- ASYNC operations start immediately and do not block execution
- BLOCKING operations start only when the previous one is finished

------------------------
GLOBAL RULES
------------------------

- One instruction per line
- Use only the operations listed above
- Do not invent new operations
- Do not invent new arguments
- Do not mix FAIL with other operations
- The program must represent a valid executable plan

========================
INVALID OUTPUT EXAMPLE
========================

ROTATE_LEFT 90
SHOOT ENEMY
MOVE FORWARD
STOP_FIRE
FAIL something
MOVE 1 0
FAIL reason
END

========================
VALID OUTPUT EXAMPLE
========================

ASYNC ROTATE_TO_TARGET MONSTER_2
MOVE 3 0
FIRE 2