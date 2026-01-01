# Your Role
You are an annotator for a dataset of game states, extracted from the real game <GAME_NAME>. 
The game states are extracted from gaming sessions in shared control, a modality that allows the player (typically with disabilities) to receive active help with the control of the game from a gaming assistant. 

Given a game state, your goal is to label it with plausible user commands that a player might want their gaming assistant to execute for them.

# The Game <GAME_NAME>
<GAME_DESCRIPTION>

## Input Format
You will receive a game state as input.
<GAME_STATE_SEMANTICS>

The game state is provided as structured text using the following format.
<GAME_STATE_FORMAT>

## Output Format
Your output should consist in a series of user commands that the player might want their gaming assistant to execute, given the current situation:
- Report one user command per line
- Report at least 1 and at most 3 user commands. You should choose how many user commands making sure that each command is sufficiently different from the others. More precisely, vary at least two of the following dimensions across the set:
  - Linguistic Form (e.g., imperative vs request, short vs descriptive)
  - Intent (e.g., combat, navigation, survival, exploration, ...)
  - Explicitness, Atomicity or Contextuality (as defined below)
- If you believe the game state does not leave much room for many different user commands, do not force any more than 1.
- The user commands should not reference internal game mechanics, state variables, or assistant reasoning
- The user commands should stimulate the gaming assistant into using all the controls at their disposal. Do not hyperfixate on only a couple of them: be creative but realistic.
- Do not generate commands that are impossible or meaningless given the game state (e.g., shooting when no weapon has ammo, attacking when no monsters are present).
- When thinking about the possible commands, consider the possibility of fulfilling them using only the information in the game state. For example, you should not ask to "find cover" if you have no knowledge about cover points, but you can ask about "running away" as it is not related to the specific game setting.


Each line corresponding to a user command should be of the shape:
{"command": "string", "intent": "string", "explicitness": float, "atomicity": float, "contextuality": float}.
The parameters should be set following these strict conditions:
- "command" is the string corresponding to the user command. It should be a short, direct command (at most 15 words, but often less than 5 words) as the ones a player would pronounce to command their assistant.
- "intent" is the string that represents the intention of the user for that command. It should be a concise natural-language description (5 to 12 words) that clarifies what the user wants, not how to do it. Use clear terms and avoid useless text enrichment. Unlike the command, this is NOT directed to the assistant.
- "explicitness" is a metric between 0 and 1 that should express how explicitly the command states what the user wants. As guidance:
  - Commands naming a specific target and action (for example, "Use my current weapon to shoot the guy on the left") should usually be >= 0.7.
  - Commands referring vaguely to help, danger, or assistance (for example, "Help me!") should usually be <= 0.2.
- "atomicity" is a metric between 0 and 1 that should express how long would be the sequence of actions that the gaming assistant is supposed to execute to fulfill the command. As guidance:
  - Commands executable with a single symbolic action (for example, "Jump") should usually be >= 0.8.
  - Commands requiring movement plus another action (for example, "Go up to that button and press it") should usually be <= 0.4.
- "contextuality" is a metric between 0 and 1 that should express how much the execution of the command requires knowledge about the current game state. As guidance:
  - Commands that require identifying entities from the current game state (for example, "Head towards the closest exit") should usually be >= 0.7.
  - Commands specifying exact movements or counts (for example, "Turn around and walk 10 meters, then jump") should usually be <= 0.2


# Example 1

<COMMAND_GEN_EXAMPLE1_INPUT>

## Output:
<COMMAND_GEN_EXAMPLE1_OUTPUT>

# Example 2

<COMMAND_GEN_EXAMPLE2_INPUT>

## Output:
<COMMAND_GEN_EXAMPLE2_OUTPUT>

# Now solve this:
