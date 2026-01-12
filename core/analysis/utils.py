# Evaluation functions
import Levenshtein
from collections import Counter

from core.utils.types import GameAction


def exact_match(teacher: str, student: str) -> bool:
    """Exact match between teacher and student"""
    return teacher.strip() == student.strip()

def normalized_edit_distance(teacher: str, student: str) -> float:
    """Normalized character-level Levenshtein distance in [0, 1]."""
    t = teacher.strip()
    s = student.strip()

    if not t and not s:
        return 0.0
    if not t or not s:
        return 1.0

    return Levenshtein.distance(t, s) / max(len(t), len(s))

def lcs_action_type_f1(
    teacher_actions: list[str],
    student_actions: list[str],
) -> float:
    """F1-Score computed on the types of actions in the Longest Common Subsequence"""
    t, s = teacher_actions, student_actions
    if not t and not s:
        return 1.0
    if not t or not s:
        return 0.0

    n, m = len(t), len(s)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if t[i - 1] == s[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[n][m]

    precision = lcs / len(s)
    recall = lcs / len(t)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)

def action_type_f1(
    teacher_actions: list[str],
    student_actions: list[str],
) -> float:
    """F1-Score computed on the types of actions between student and teacher"""
    t = Counter(teacher_actions)
    s = Counter(student_actions)

    if not t and not s:
        return 1.0
    if not t or not s:
        return 0.0

    tp = sum(min(t[a], s[a]) for a in t.keys() | s.keys())

    precision = tp / sum(s.values())
    recall = tp / sum(t.values())

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)

