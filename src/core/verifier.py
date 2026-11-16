"""
Verification of the answer.
"""

from abc import abstractclass
from unicodedata import ucd_3_2_0

@abstractclass
class AnswerVerifier:
    def __init__(self):
        """
        Compares policy response to reference.
        """

class RewardFunction(AnswerVerifier):
    def __init__(self):
        """
        Assigning training reward.
        """
        pass
