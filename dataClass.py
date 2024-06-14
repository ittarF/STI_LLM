import pandas as pd
import json
from prompts import generate_NER_prompt, generate_CEA_prompt, generate_tableDesc_prompt, generate_CEA_prompt_with_t_desc
import requests
import os
import re
from dotenv import load_dotenv
load_dotenv()


class DataTable:
    def __init__(self, file_path):
        self.file_path = file_path
        self.name = self.file_path.split('/')[-1].strip('.csv').strip('.json')
        self.ner = None
        self.cea = None
        self.t_desc = None
        self.cta = None
        if file_path.endswith('.csv'):
            self.data = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            self.data = pd.read_json(file_path)
        else:
            raise ValueError("Unsupported file format. Please use a CSV or JSON file.")

        # Convert all cells to lowercase
        self.data = self.data.map(lambda x: x.lower() if isinstance(x, str) else x)

    def get_column(self, column_index):
        if 0 <= column_index < len(self.data.columns):
            return self.data.iloc[:, column_index]
        else:
            raise IndexError(f"Column index '{column_index}' is out of range.")

    def get_row(self, row_index):
        if 0 <= row_index < len(self.data):
            return self.data.iloc[row_index]
        else:
            raise IndexError("Row index out of range.")

    def get_cell(self, row_index, column_index):
        if not (0 <= column_index < len(self.data.columns)):
            raise IndexError(f"Column index '{column_index}' is out of range.")
        if not (0 <= row_index < len(self.data)):
            raise IndexError("Row index out of range.")
        return self.data.iat[row_index, column_index]
    
    def generate_ner_labels(self, llm):
        if(self.ner==None):
            prompt = generate_NER_prompt(self.data)
            out = llm.invoke(prompt)
            self.ner = json.loads(out.content.replace("'", '"'))
        else:
            print("Labels already generated.\n")
        return self.ner      
    
    def generate_t_description(self, llm):
        if (self.t_desc == None):
            prompt = generate_tableDesc_prompt(self.data)
            out = llm.invoke(prompt)
            match = re.search(r'(\{.*\})', out.content, re.DOTALL)
            if match:
                dict_str = match.group(1)
                self.t_desc = json.loads(dict_str)
            else:
                print("Error in parsing the ouput")
                self.t_desc = out.content
        return self.t_desc
           
    def generate_cea_annotatons(self, llm):
        cea_dict = {}
        if (self.ner == None):
            return 'Perform Columns Named Entities classification first!'
        if (self.t_desc == None):
            return 'Table description is missing!'
        if (self.cea == None):
            ner_columns_idx = [idx for idx in range(len(self.ner)) if self.ner[str(idx)] == 'NEC']
            for j in ner_columns_idx:
                for i in range(len(self.data)):
                
                    cell_content = self.data.iloc[i, j]
                    entity_retrieval = LamAPI(cell_content) 
                    prompt = generate_CEA_prompt_with_t_desc(self.data, cell_content, entity_retrieval, self.t_desc)
                    out = llm.invoke(prompt)
                    match = re.search(r'\[\[\[(.*?)\]\]\]', out.content)

                    if match:
                        id = match.group(1)
                    else:
                        # If no match is found, try to match the first occurrence of 'Q' followed by numbers
                        match = re.search(r'\bQ\d+\b', out.content)
                        if match:
                            id = match.group(0)
                        else:
                            print('No match found')
                            id = None
                    
                    cea_dict[str((i,j))] = {'id': id, 'llm_output': out.content}
            self.cea = cea_dict
        else:
            print('CEA annotations already generated.')
        return cea_dict        
        
    def save_json(self, folder):
        dict = {
            "name": self.name,
            "description": self.t_desc,
            "named_entity_columns": self.ner,
            "cea": self.cea
        }
        # Serialize data into file:
        json.dump(dict, open(folder + f"/{self.name}.json", 'w'))
        #print(f"Saved as: {folder}/{self.name}.json")
        return  
            

def LamAPI(cell_content):
    
    url = 'https://lamapi.hel.sintef.cloud/lookup/entity-retrieval'
    params = {
        'name': f'{cell_content}',
        'token': os.getenv("LAMAPI_KEY"),
        'kg': 'wikidata',
        'fuzzy': 'True'
    }
    headers = {'accept': 'application/json'}

    response = requests.get(url, params=params, headers=headers)

    if response.status_code == 200:
        data = response.json()
        # Process the JSON data here
    else:
        print("Error:", response.status_code)
    
    list_of_dicts = data[f'{cell_content}']
    keys_to_select = ['id', 'name', 'types']

    selected_dicts = [{k: v for k, v in d.items() if k in keys_to_select} for d in list_of_dicts]
    res_dict = {d['id']: d for d in selected_dicts}
    
    for item in selected_dicts:
        for type_dict in item['types']:
            if 'id' in type_dict:
                del type_dict['id']

    string = json.dumps(selected_dicts)
    string = string.replace('"', '').replace('name:', '').replace('  ', ' ').replace('[', '').replace(']', '')
    
    return string

