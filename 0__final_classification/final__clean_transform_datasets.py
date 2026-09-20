import json
import re
import random
import os
from pathlib import Path

import re
import unicodedata
import emoji
import pandas as pd
from bs4 import BeautifulSoup
import pandas as pd 
import numpy as np
from typing import Any

# NaN like strings 
MISSING_TEXT_VALUES = {"", "none", "nan", "null", "<na>", "n/a"}

# URL pattern
URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+",
    flags=re.IGNORECASE,
)

# Simple check for HTML tags
HTML_PATTERN = re.compile(r"<[^>]+>")

# LaTeX environment tags such as \begin{equation}, \end{align}
LATEX_ENV_PATTERN = re.compile(
    r"\\(?:begin|end)\s*\{[^{}]*\}",
    flags=re.IGNORECASE,
)

# LaTeX commands such as \frac, \sqrt, \alpha, \text
LATEX_COMMAND_PATTERN = re.compile(
    r"\\[a-zA-Z]+(?:\s*\[[^\]]*\])?"
)

# Check if the whole line is of the form $...$ or $$...$$
DOLLAR_ONLY_PATTERN = re.compile(
    r"""
    ^\s*
    (?:
        \$\$.*?\$\$       # $$ ... $$
        |
        \$[^$]*?\$        # $ ... $
    )
    \s*$
    """,
    flags=re.DOTALL | re.VERBOSE,
)

# Remove LaTeX symbols inside a string
LATEX_SYMBOL_PATTERN = re.compile(
    r"\$\$?|"           # $, $$
    r"\\[\[\]()]|"      # \(, \), \[, \]
    r"[{}^_&]"
)

# Remove all except English letters, numbers, spaces, and basic punctuation
NON_TEXT_PATTERN = re.compile(
    r"""[^a-z0-9\s.,!?'"-]"""
)

# Consecutive whitespace
WHITESPACE_PATTERN = re.compile(r"\s+")

def normalize_text(content) -> str:
    # Handle abnormal objects like list, dict, set, tuple
    if isinstance(content, (list, dict, set, tuple)):
        return ""

    # Handle None, NaN
    if content is None:
        return ""

    try:
        if pd.isna(content):
            return ""
    except (TypeError, ValueError):
        return ""

    # Normalize Unicode first; convert characters to their standard forms
    text = str(content)

    # Catch values that become missing-like after normalization
    if text.lower() in MISSING_TEXT_VALUES:
        return ""

    # Normalize Unicode compatibility characters (e.g. convert full-width/half-width letters to standard form)
    text = unicodedata.normalize("NFKC", text)

    # Remove URLs before BeautifulSoup to avoid treating URLs as HTML
    text = URL_PATTERN.sub(" ", text)

    # Use BeautifulSoup only if there are actual HTML tags
    if HTML_PATTERN.search(text):
        text = BeautifulSoup(text, "html.parser").get_text(separator=" ")

    # Remove emojis
    text = emoji.replace_emoji(text, replace=" ")

    # Lowercase
    text = text.lower()

    # If the entire line is a $...$ or $$...$$ equation, return empty string. Must check this before removing $ symbols
    if DOLLAR_ONLY_PATTERN.fullmatch(text):
        return ""

    # Remove LaTeX environments and commands
    text = LATEX_ENV_PATTERN.sub(" ", text)
    text = LATEX_COMMAND_PATTERN.sub(" ", text)

    # Remove LaTeX symbols
    text = LATEX_SYMBOL_PATTERN.sub(" ", text)

    # Remove any character except English letters, numbers, and basic punctuation
    text = NON_TEXT_PATTERN.sub(" ", text)

    # Replace consecutive whitespace with a single space
    text = WHITESPACE_PATTERN.sub(" ", text).strip()

    return text
    

############################################################################## 
# Below are the functions to process StepVerify dataset
############################################################################## 
def get_tutor_turn_only_utterances_stepverify(data: list[dict[str, Any]]) -> pd.DataFrame:
    examples = []
    for row in data:
        turns = row.get("dialog_history", []) 

        for turn in turns:
            # ignore the student turn
            if turn.get("user").strip().lower() != "teacher":
                continue

            pedagogy_label = turn.get("pedagogy") 

            # ignore teacher turns without pedagogy label
            if pedagogy_label is None:
                continue 

            text = normalize_text(turn.get("text", ""))

            # ignore empty text 
            if not text:
                continue 

            examples.append({
                "text": text,
                "label": pedagogy_label,
            })

    tutor_turn_only_utterances_df = pd.DataFrame(examples)

    return tutor_turn_only_utterances_df



def get_all_turn_utterances_stepverify(data: list[dict[str, Any]], include_speaker: bool = True) -> dict | None:
    examples = [] 
    for row in data:
        # dialog history with student and teacher utterance turns
        turns = row.get("dialog_history", [])
        
        if not turns:
            return None

        # get the classification target turn: the last teacher turn (utterance & pedagogy label)
        target_turn = turns[-1]

        if target_turn.get("user").strip().lower() != "teacher":
            return None

        # if the last turn of the each utterance is a teacher, get the teacher turn label
        label = target_turn.get("pedagogy")

        if label is None:
            return None 

        # serialize the dialog history with student and teacher utterance turns 
        parts = []

        for turn in turns:
            text = normalize_text(turn.get("text", ""))

            if not text:
                continue

            # if include_speaker is True, add teacher and student markers to the text 
            if include_speaker: 
                speaker = turn.get("user", "Unknown")

                if speaker.strip().lower() == "teacher":
                    text = f"T: {text}"
                elif speaker.strip().lower() == "student":
                    text = f"S: {text}"
                else:
                    return None  

            parts.append(text)

        if not parts:
            return None  

        # add </s> token between each utterance turn  
        serialzed_example = {
            "text": " </s> ".join(parts),
            "label": label,
        }
        
        examples.append(serialzed_example) 

    all_turn_utterances_df = pd.DataFrame(examples)

    return all_turn_utterances_df



def get_preceding_n_utterance_turns_stepverify(data: list[dict[str, Any]], n: int, include_speaker: bool = True) -> dict | None:
    if n < 1:
        raise ValueError("n must be at least 1")

    examples = []
    for row in data:
        # dialog history with student and teacher utterance turns
        turns = row.get("dialog_history", [])
        
        if not turns:
            return None

        # get the classification target turn: the last teacher turn (utterance & pedagogy label)
        target_turn = turns[-1]

        if target_turn.get("user").strip().lower() != "teacher":
            return None

        # if the last turn of the each utterance is a teacher, get the teacher turn label
        label = target_turn.get("pedagogy")

        if label is None:
            return None 

        # the dialog history that ends with a teacher turn utterance with a pedagogy label 
        selected_turns = turns 

        # the index of the last teacher turn utterance with a pedagogy label 
        target_idx = None

        for i in range(len(selected_turns) - 1, -1, -1): 
            turn = selected_turns[i] 

            if turn.get("user").strip().lower() == "teacher" and turn.get("pedagogy") is not None:
                target_idx = i 
                break 

        if target_idx is None:
            return None 

        label = turns[target_idx].get("pedagogy") 

        # container to store the teacher utterance with preceding n utterance turns 
        preceding_turns = [] 

        for i in range(target_idx, -1, -1):
            turn = selected_turns[i] 
            text = normalize_text(turn.get("text", ""))

            if not text:
                continue

            if include_speaker:
                speaker = turn.get("user", "Unknown")
                if speaker.strip().lower() == "teacher":
                    text = f"T: {text}"
                elif speaker.strip().lower() == "student":
                    text = f"S: {text}"
                else:
                    return None
                    
            preceding_turns.append(text) 

            # n preceding turns + the target teacher turn utterance (1 turn) = n + 1 turns 
            if len(preceding_turns) == n + 1:
                break 

            
        if not preceding_turns:
            return None 
        elif len(preceding_turns) != n + 1:
            return None  

        # reverse the preceding turns to match the order of the turns in the dialog history 
        preceding_turns.reverse()


        serialzed_example = {
            "text": " </s> ".join(preceding_turns),
            "label": label,
        }

        examples.append(serialzed_example) 

    preceding_n_utterance_turns_df = pd.DataFrame(examples)

    return preceding_n_utterance_turns_df



############################################################################## 
# Below are the functinos to process EEDI and MathtutorMR datasets
############################################################################## 

def clean_EEDI_MathtutorMR(df) -> pd.DataFrame:  
    # normalize the content of the dataset
    df["content"] = df["content"].map(normalize_text)

    # delete rows with empty content
    df = df.loc[df["content"].str.len().gt(0)].reset_index(drop=True)

    # reassign sequence numbers to the remaining rows 
    df["new_message_sequence"] = df.groupby("conversation_id", sort=False).cumcount().add(1) 

    return df


# MathTutorMR dataset contains rows with .close message, which is not an utterance that would appear in the real conversation.
def delete_MathtutorMR_rows_with_close(mathtutorMR_df: pd.DataFrame) -> pd.DataFrame:
    # delete rows with close message
    mathtutorMR_df = mathtutorMR_df.loc[~mathtutorMR_df["content"].str.contains(r"\.close", case=False, na=False)].reset_index(drop=True)
    return mathtutorMR_df


def join_content(values):
    return " ".join(values) 

def compress_EEDI_MathtutorMR(df) -> pd.DataFrame:
    work_df = df.copy() 

    # sort the dataframe by conversation id and message sequence number
    work_df = df.sort_values(["conversation_id", "message_sequence"], kind="stable", ignore_index=True).copy()

    # convert is_tutor value to string and fill missing values with empty string
    work_df["is_tutor"] = work_df["is_tutor"].astype(str).fillna("")

    # fill missing values with empty string and strip whitespace
    work_df["content"] = work_df["content"].fillna("").astype(str).str.strip()

    # check if the conversation id is the same as the previous conversation id
    same_conversation = work_df["conversation_id"].eq(work_df["conversation_id"].shift())

    # check if the is_tutor value is the same as the previous is_tutor value
    same_speaker = work_df["is_tutor"].eq(work_df["is_tutor"].shift())

    # if preceding turn message and the speaker are the same, then group the messages together
    work_df["_speaker_run"] = (~(same_conversation & same_speaker)).cumsum() 

    compressed_df = (
        work_df
        .groupby(
            ["conversation_id", "_speaker_run"],
            sort=False,
            observed=True
        )
        .agg(
            # student=("student", "first"),
            is_tutor=("is_tutor", "first"),
            content=("content", join_content),
            message_sequence=("message_sequence", "first"),
        )
        .reset_index()
        .drop(columns="_speaker_run")
    )

    compressed_df["new_message_sequence"] = compressed_df.groupby("conversation_id", sort=False).cumcount().add(1) 

    return compressed_df


def filter_compressed_turns_EEDI_MathmentorMR(df: pd.DataFrame) -> pd.DataFrame:
    """
        For conversation text in each conversation id group, the last turn should be a teacher turn ( is_tutor == 1 ).
        If the last turn is not a teacher turn, delete the last turn in the same conversation id group. 
        After processing, the number of turns in each conversation id group should be at least 3 turns. 
    """
    work_df = df.copy() 

    # convert is_tutor value to numeric
    work_df["is_tutor"] = pd.to_numeric(work_df["is_tutor"], errors="coerce")
    
    # check if the last row in each conversation group 
    last_row = work_df.groupby("conversation_id").cumcount() == (work_df.groupby("conversation_id")["conversation_id"].transform("count") - 1)

    # mask the rows to delete ( last row in each conversation group, which is a student turn )
    drop_mask = last_row & (work_df["is_tutor"] == 0) 

    work_df = work_df.loc[~drop_mask].copy() 

    # check if the conversation has alternatig teacher and student turns 
    conversation_group = work_df.groupby("conversation_id") 
    previous_speaker = conversation_group["is_tutor"].shift()
    alternating_turns = previous_speaker.isna() | work_df["is_tutor"].ne(previous_speaker)                      # first row or the current speaker is different from the previous speaker 
    alternating_turns_mask = alternating_turns.groupby(work_df["conversation_id"], sort=False).transform("all") # check if all turns in the conversation are alternating 

    # check if the last turn in each conversation group is a teacher turn 
    conversation_with_last_tutor_turn_mask = work_df.groupby("conversation_id")["is_tutor"].transform("last").eq(1) 

    # check the length of the turns in each conversation group
    rows_with_at_least_3_turns_mask = work_df.groupby("conversation_id")["conversation_id"].transform("count").ge(3) 

    # keep conversation groups that meet the criteria
    work_df = work_df.loc[conversation_with_last_tutor_turn_mask & rows_with_at_least_3_turns_mask & alternating_turns_mask].copy()

    # assign new message sequence number 
    work_df["new_message_sequence"] = work_df.groupby("conversation_id", sort=False).cumcount().add(1) 

    return work_df 

    
def get_all_turn__preceding_n_turn__utterances_EEDI_MathmentorMR(df: pd.DataFrame, include_speaker: bool = False, n: int = None) -> pd.DataFrame:
    work_df = df.copy() 

    # convert is_tutor value to numeric
    work_df["is_tutor"] = pd.to_numeric(work_df["is_tutor"], errors="coerce")
    
    # sort the dataframe by conversation id and new message sequence number
    work_df = work_df.sort_values(["conversation_id", "new_message_sequence"], kind="stable", ignore_index=True).copy()

    # Handle possible missing values and whitespace in the 'content' column to avoid errors
    work_df["content"] = work_df["content"].fillna("").astype(str).str.strip()      

    # convert to boolean series where True is tutor, False is student
    is_teacher = work_df["is_tutor"].eq(1)              

    # if include_speaker is True, add speaker info to the turn 
    if include_speaker: 
        work_df["turn"] = np.where(is_teacher, "T: " + work_df["content"], "S: " + work_df["content"])
    else:
        work_df["turn"] = work_df["content"] 

    
    # if n is not None, get the preceding n turns + the last tutor turn utterance (1 turn) = n + 1 turns  
    if n is not None:
        if n < 1:
            raise ValueError("n must be at least 1")
        work_df = ( 
            work_df
            .sort_values(["conversation_id", "new_message_sequence"], kind="stable", ignore_index=True)
            .groupby("conversation_id", sort=False)
            .tail(n+1) 
        )

    # add special tokens to set boundaries between speakers 
    work_df = work_df.groupby("conversation_id", sort=False, observed=True)["turn"].agg(" </s>".join).rename("text").reset_index()

    # return only the text column
    work_df = work_df[["text"]]

    return work_df 
    
    
def get_tutor_turn_only_EEDI_MathmentorMR(df: pd.DataFrame) -> pd.DataFrame:
    work_df = df.copy() 
    
    # convert is_tutor value to numeric
    work_df["is_tutor"] = pd.to_numeric(work_df["is_tutor"], errors="coerce")
    
    # sort the dataframe by conversation id and new message sequence number
    work_df = work_df.sort_values(["conversation_id", "new_message_sequence"], kind="stable", ignore_index=True).copy()

    # keep only the tutor turns 
    work_df = work_df[work_df["is_tutor"] == 1]

    # remove possible missing values and whitespace in the 'content' column to avoid errors
    work_df["content"] = work_df["content"].fillna("").astype(str).str.strip()

    # remove possible duplicate content
    work_df = work_df.drop_duplicates(subset=["content"], keep="first").reset_index(drop=True)

    # rename the content column to text 
    work_df = work_df.rename(columns={"content": "text"})

    # return only the text column
    work_df = work_df[["text"]]
    
    return work_df 



    
