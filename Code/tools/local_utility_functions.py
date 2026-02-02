from pathlib import Path
import pandas as pd
import numpy as np

def adapt_txt(file_path: Path, save_path: Path) -> pd.DataFrame:
    """
    Reads a .txt file and adapts it into a pandas DataFrame.
    
    Parameters:
    - file_path (Path): The path to the .txt file.
    
    Returns:
    - pd.DataFrame: The adapted DataFrame.
    """
    # Load test.txt file
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    lines_clean = [line for line in lines if not line.startswith('>')]
    cleaned_content = ''.join(lines_clean)
    cleaned_content = cleaned_content.replace('data =', '', 1).strip()
    cleaned_content = cleaned_content.replace("'", '"')
    inline_content = cleaned_content.replace(" ", '').replace("\n",'')
    inline_content = eval(inline_content)
    
    df = pd.DataFrame.from_dict(inline_content)
    if save_path is not None and save_path.suffix == '.csv':
        df.to_csv(save_path, index=False)
    
    return df

def remove_polish_characters(text: str) -> str:
    """
    Removes Polish special characters from a given text string.
    
    Parameters:
    - text (str): The input text string.
    
    Returns:
    - str: The text string with Polish characters replaced by their non-accented counterparts.
    """
    polish_chars = {
        'ą': 'a', 'ć': 'c', 'ę': 'e', 'ł': 'l', 
        'ń': 'n', 'ó': 'o', 'ś': 's', 'ź': 'z', 
        'ż': 'z', 'Ą': 'A', 'Ć': 'C', 'Ę': 'E', 
        'Ł': 'L', 'Ń': 'N', 'Ó': 'O', 'Ś': 'S', 
        'Ź': 'Z',  'Ż': 'Z'
    }
    
    for pol_char, repl_char in polish_chars.items():
        text = text.replace(pol_char, repl_char)
    
    return text

def nuts_code_to_teryt(nuts_code: str) -> str:
    """
    Converts a NUTS code to a TERYT code by removing the first character.
    
    Parameters:
    - nuts_code (str): The NUTS code.
    
    Returns:
    - str: The corresponding TERYT code.
    """
    
    '''if len(nuts_code) == 11:
        k = 2
    elif len(nuts_code) == 12:
        k = 3
    else:
        raise ValueError("NUTS code must be either 11 or 12 characters long.")'''
    k=3
    
    # Code of gmina - the last 2 digits of NUTS code
    gmina_id = nuts_code[-k:]
    # Code of powiat - the 2 digits before gmina code
    powiat_id = nuts_code[-(k+2):-k]
    # Code of voivodeship
    voivodeship_id = nuts_code[1:3]
    
    teryt_code = voivodeship_id + powiat_id + gmina_id
    return teryt_code
    
    