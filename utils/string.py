
import re
from typing import Optional, Sequence, Tuple

ENDOFSENTENCE_PATTERN_STR = r"""
    # Western punctuation
    (?<![A-Z])           # Not preceded by uppercase letter (e.g., "U.S.A.")
    (?<!\d\.\d)          # Not preceded by decimal (e.g., "3.14")
    (?<!^\d\.)           # Not preceded by numbered list (e.g., "1. Start")
    (?<!\d\s[ap])        # Not preceded by time (e.g., "3 p.m.")
    (?<!Mr|Ms|Dr)        # Not preceded by titles
    (?<!Mrs)
    (?<!Prof)
    [\.\?\!;]|           # Western sentence ending punctuation
    
    # Chinese/full-width punctuation
    [。？！；]|           # Chinese sentence ending punctuation
    
    # End of string is also end of sentence
    $
"""

ENDOFSENTENCE_PATTERN = re.compile(ENDOFSENTENCE_PATTERN_STR, re.VERBOSE)

EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

NUMBER_PATTERN = re.compile(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?")

StartEndTags = Tuple[str, str]


def replace_match(text: str, match: re.Match, old: str, new: str) -> str:
    """Replace occurrences of a substring within a matched section of a given
    text.

    Args:
        text (str): The input text in which replacements will be made.
        match (re.Match): A regex match object representing the section of text to modify.
        old (str): The substring to be replaced.
        new (str): The substring to replace `old` with.

    Returns:
        str: The modified text with the specified replacements made within the matched section.

    """
    start = match.start()
    end = match.end()
    replacement = text[start:end].replace(old, new)
    text = text[:start] + replacement + text[end:]
    return text


def match_endofsentence(text: str) -> int:
    """Finds the position of the end of a sentence in the provided text string.

    This function processes the input text by replacing periods in email
    addresses and numbers with ampersands to prevent them from being
    misidentified as sentence terminals. It then searches for the end of a
    sentence.

    Supports both English and Chinese punctuation marks as sentence terminators.

    Args:
        text (str): The input text in which to find the end of the sentence.

    Returns:
        int: The position of the end of the sentence if found, otherwise 0.
    """
    if not text:
        return 0
    
    text = text.rstrip()

    # Replace email dots by ampersands to avoid false sentence endings
    emails = list(EMAIL_PATTERN.finditer(text))
    for email_match in emails:
        text = replace_match(text, email_match, ".", "&")

    # Replace number dots by ampersands to avoid false sentence endings
    numbers = list(NUMBER_PATTERN.finditer(text))
    for number_match in numbers:
        text = replace_match(text, number_match, ".", "&")

    # Define sentence ending punctuation
    english_punct = {'.', '?', '!', ';'}
    chinese_punct = {'。', '？', '！', '；'}
    all_punct = english_punct.union(chinese_punct)
    
    # Look for sentence endings
    for i in range(len(text)):
        if text[i] in all_punct:
            # Skip common abbreviations with period
            if text[i] == '.':
                if i > 0:
                    # Check for common title abbreviations
                    prev_word = text[:i].split()[-1].lower() if text[:i].split() else ""
                    if prev_word in ["mr", "mrs", "ms", "dr", "prof"]:
                        continue
                    # Skip decimal numbers
                    if i > 0 and i < len(text) - 1:
                        if text[i-1].isdigit() and text[i+1].isdigit():
                            continue
            
            # We've found a valid sentence ending
            return i + 1
    
    # No sentence ending found
    return 0
  
  
def parse_start_end_tags(
    text: str,
    tags: Sequence[StartEndTags],
    current_tag: Optional[StartEndTags],
    current_tag_index: int,
) -> Tuple[Optional[StartEndTags], int]:
    """Parses the given text to identify a pair of start/end tags.

    If a start tag was previously found (i.e. current_tags is valid), wait for
    the corresponding end tag.  Otherwise, wait for a start tag.

    This function will return the index in the text that we should start parsing
    in the next call and the current or new tags.

    Parameters:
    - text (str): The text to be parsed.
    - tags (Sequence[StartEndTags]): List of tuples containing start and end tags.
    - current_tags (Optional[StartEndTags]): The currently active tags, if any.
    - current_tags_index (int): The current index in the text.

    Returns:
    Tuple[Optional[StartEndTags], int]: A tuple containing None or the current
    tag and the index of the text.

    """
    # If we are already inside a tag, check if the end tag is in the text.
    if current_tag:
        _, end_tag = current_tag
        if end_tag in text[current_tag_index:]:
            return (None, len(text))
        return (current_tag, current_tag_index)

    # Check if any start tag appears in the text
    for start_tag, end_tag in tags:
        start_tag_count = text[current_tag_index:].count(start_tag)
        end_tag_count = text[current_tag_index:].count(end_tag)
        if start_tag_count == 0 and end_tag_count == 0:
            return (None, current_tag_index)
        elif start_tag_count > end_tag_count:
            return ((start_tag, end_tag), len(text))
        elif start_tag_count == end_tag_count:
            return (None, len(text))

    return (None, current_tag_index)
