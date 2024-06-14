import os
from dotenv import load_dotenv
from dataClass import DataTable
from langchain_mistralai import ChatMistralAI
import json
from tqdm import tqdm
def list_files_in_folder(folder_path):
    try:
        # Get a list of all entries in the directory
        entries = os.listdir(folder_path)
        
        # Filter out the files from the entries
        files = [os.path.join(folder_path, entry) for entry in entries if os.path.isfile(os.path.join(folder_path, entry))]
        
        return files
    except FileNotFoundError:
        return f"The folder '{folder_path}' does not exist."
    except Exception as e:
        return f"An error occurred: {e}"

def main():
    load_dotenv()
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    model = "open-mixtral-8x7b"
    llm = ChatMistralAI(model=model, temperature=0, api_key=mistral_api_key)
    tables_path = 'data/HardTablesR1/DataSets/HardTablesR1/Valid/tables'
    tables = list_files_in_folder(tables_path)
    print(f"\nNumber of tables: {len(tables)}\n")
        
    i = 0
    results = {}
    for table in tqdm(tables):
        t = DataTable(table)
        #print(f"Table name: {t.name}\n")
        #print(f"Table shape: {t.data.shape}\n")
        t.generate_t_description(llm)
        t.generate_ner_labels(llm)
        t.generate_cea_annotatons(llm)
        results[t.name] = {
                            'nec': t.ner,
                            'cea': t.cea
        }
        t.save_json('results/HardTablesR1/Valid/tables')
        i += 1
        if i > 1:
            json.dump(results, open('results/HardTablesR1/Valid/CEA/Valid.json', 'w'))
            #break
   
    json.dump(results, open('results/HardTablesR1/Valid/CEA/Valid.json', 'w'))

if __name__ == "__main__":
    main()
    
