def get_table_str(df):
    column_names = df.columns.tolist()
    table_str = "col: "
    table_str += "| " + " | ".join(column_names) + " | "
    for index, row in df.iterrows():
        row_str = " | " + " | ".join(str(row[col]) for col in column_names) + " | "
        table_str += f"[SEP] col {index + 1}: {row_str}"
    return table_str

def candidates_as_str(candidates):
    
    list_of_candidates = ""
    for c in candidates:
        if c['description'] == '':
            c['description'] = 'None'
        list_of_candidates += f"<[ID] {c['id']} [NAME] {c['name']} [DESC] {c['description']} [TYPE] {c['types'][0]['name']}>, "
    
    return list_of_candidates[:-2]

def build_prompt(table_str, column_name, cell_content, candidates, t_desc):
    TASK = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
    INSTRUCTION = "### Instruction: This is an entity linking task. The goal for this task is to link the selected entity mention in the table cells to the entity in the knowledge base. You will be given a list of referent entities, with each one composed of an entity id, name, its description and its type. Please choose the correct one from the referent entity candidates. Note that the Wikipedia page, Wikipedia section and table caption (if any) provide important information for choosing the correct referent entity."
    INPUT = f"### Input: [TLE] {t_desc} [TAB] {table_str}"
    QUESTION = f"### Question: The selected entity mention in the table cell is: {cell_content}. The column name for ’{cell_content}’ is {column_name}. "
    CANDIDATES = f"The referent entity candidates are: {candidates}"
    tablellama_prompt = (
        f"{TASK}\n\n"
        f"{INSTRUCTION}\n\n"
        f"{INPUT}\n\n"
        f"{QUESTION}"
        f"{CANDIDATES}. \nIf there are no candidates that matched the cell content the response is <NIL>. What is the correct referent entity for the entity mention ’{cell_content}’ ?\n\n"  
        "### Response: "
    )
    return tablellama_prompt