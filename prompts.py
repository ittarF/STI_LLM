import pandas as pd

def generate_NER_prompt(df):
    task_description = "You are to classify each column in a given table as either a Named Entity Column (NEC) or a Literal Column (LC)."
    definitions = (
        "Definitions:\n"
        "- Named Entity Column (NEC): Columns that contain names of people, organizations, locations, or other proper nouns.\n"
        "- Literal Column (LC): Columns that contain numerical values, dates, measurements, or other literal values."
    )
    examples = (
        "Examples:\n"
        "- Named Entity Column (NEC) Examples:\n"
        '  - Column with values: ["John Doe", "Jane Smith", "Company XYZ", "Paris"]\n'
        '  - Column with values: ["Microsoft", "Apple", "Google", "Amazon"]\n\n'
        "- Literal Column (LC) Examples:\n"
        '  - Column with values: [34, 56, 78, 23]\n'
        '  - Column with values: ["2021-01-01", "2022-05-12", "2023-08-23"]\n'
        '  - Column with values: [5.6, 3.4, 2.8, 4.5]'
    )
    
    # Extract column names
    column_names = df.columns.tolist()
    
    # Construct the table for display in the prompt
    table_str = "Table for Classification:\n"
    table_str += "| " + " | ".join(column_names) + " |\n"
    table_str += "|-" + "-|-".join(["-" * len(col) for col in column_names]) + "-|\n"
    for index, row in df.iterrows():
        row_str = "| " + " | ".join(str(row[col]) for col in column_names) + " |\n"
        table_str += row_str
    
    # Add classification request
    classification_request = "Classification Request:\nBased on the above definitions and examples, please classify each column in the provided table as either Named Entity Column (NEC) or Literal Column (LC).\nPlease provide the response strictly in the format {'column indexs': 'classification'}. Do not include any additional text or explanation.\nExample: {'0': 'NEC', '1': 'LC', '2': NEC} \nWrong Answer: {'col0': 'NEC', 'col1': 'LC', 'col2': NEC}. keys must be string of integers"
    
    classification_str = "Classification:\n"
    #for i, col in enumerate(column_names, start=1):
    #    classification_str += f"{i}. {col}: [Your classification]\n"
    
    # Combine all parts to form the final prompt
    prompt = (
        f"{task_description}\n\n"
        f"{definitions}\n\n"
        f"{examples}\n\n"
        f"{table_str}\n\n"
        f"{classification_request}\n\n"
        f"{classification_str}"
    )
    
    return prompt

def generate_CEA_prompt(df, cell_content, ER):
    '''
    input:
        df: pandas dataframe
        cell_content : str
        ER: retrieved entities from lamapi
    '''
    task_description = "You are to choose which retrieved entity is the correct entity to be associated to the cell content"
    # Extract column names
    column_names = df.columns.tolist()
    
    # Construct the table for display in the prompt
    table_str = "Table:\n"
    table_str += "| " + " | ".join(column_names) + " |\n"
    table_str += "|-" + "-|-".join(["-" * len(col) for col in column_names]) + "-|\n"
    for index, row in df.iterrows():
        row_str = "| " + " | ".join(str(row[col]) for col in column_names) + " |\n"
        table_str += row_str
    cell = f"Cell Content: {cell_content}"
    entities = f"Retrieved Entities and their types: {ER}"
    # Add classification request
    classification_request = "Classification Request:\n Study the table and the retrieved entities along with their types then associate the cell to the correct entity choosen between the list of retrieved entities.\nPlease provide the response strictly in the format {'id': 'entity_id'}. Do not include any additional text or explanation.\nExample of your answer: {'id': '12345'}"
    
    classification_str = "Chosen Entity ID:\n"
    #for i, col in enumerate(column_names, start=1):
    #    classification_str += f"{i}. {col}: [Your classification]\n"
    
    # Combine all parts to form the final prompt
    prompt = (
        f"{task_description}\n\n"
        f"{table_str}\n\n"
        f"{cell}\n\n"
        f"{ER}\n\n"
        f"{classification_request}\n\n"
        f"{classification_str}"
    )
    
    return prompt

def generate_tableDesc_prompt(df):

    task_description = 'For every column write a short description, in the format: {"0": "description of column 0", "1": "description of column 1", ..., "n": "description of column n"}.\n The description should be as restrictive and detailed as possible, identifying unique attributes that the items in the column share. If possible, include the geographical region, category, or any other specific characteristic that makes the list items distinct.'
    # Extract column names
    column_names = df.columns.tolist()
    
    # Construct the table for display in the prompt
    table_str = "Table:\n"
    table_str += "| " + " | ".join(column_names) + " |\n"
    table_str += "|-" + "-|-".join(["-" * len(col) for col in column_names]) + "-|\n"
    for index, row in df.iterrows():
        row_str = "| " + " | ".join(str(row[col]) for col in column_names) + " |\n"
        table_str += row_str

    prompt = (
        f"{task_description}\n\n"
        f"{table_str}\n\n"
    )
    
    return prompt

def generate_CEA_prompt_with_t_desc(df, cell_content, ER, description):
    '''
    input:
        df: pandas dataframe
        cell_content : str
        ER: retrieved entities from lamapi
    '''
    task_description = "Based on the table, table description, cell content and retrieved entities and their types you have to associate the cell content to one of the retrieved entities "
    # Extract column names
    column_names = df.columns.tolist()
    
    # Construct the table for display in the prompt
    table_str = "Table:\n"
    table_str += "| " + " | ".join(column_names) + " |\n"
    table_str += "|-" + "-|-".join(["-" * len(col) for col in column_names]) + "-|\n"
    for index, row in df.iterrows():
        row_str = "| " + " | ".join(str(row[col]) for col in column_names) + " |\n"
        table_str += row_str
    cell = f"Cell Content: {cell_content}"
    entities = f"Retrieved Entities and their types: {ER}"
    # Add classification request
    classification_request = "Classification Request:\n Study the table and the retrieved entities along with their types then associate the cell to the correct entity choosen between the list of retrieved entities.\nPlease provide the response strictly in the format [[[choosen_entity_id]]]. Do not include any additional text or explanation.\nExample of your answer:[[[Q89029]]]"
    
    classification_str = "Chosen Entity ID:\n"
    #for i, col in enumerate(column_names, start=1):
    #    classification_str += f"{i}. {col}: [Your classification]\n"
    
    # Combine all parts to form the final prompt
    prompt = (
        f"{task_description}\n\n"
        f"{table_str}\n\n"
        f"Table Description: {description}\n\n"
        f"{cell}\n\n"
        f"{entities}\n\n"
        f"{classification_request}\n\n"
        f"{classification_str}"
    )
    
    return prompt