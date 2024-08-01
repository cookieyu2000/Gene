import json
import pandas as pd
import requests

input_data = 'data/pub_cls.csv'
output_data = 'data/ner.txt'

def read_data(input_data):
    df = pd.read_csv(input_data, usecols=['ID'])
    return df

def call_api(id):
    api = f'https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocjson?pmids={id}'
    response = requests.get(api)
    if response.status_code == 200:
        return response.text
    else:
        return None

def process_response(response, id):
    try:
        data = json.loads(response)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON for ID {id}: {e}")
        print(f"Response content: {response}")
        return []

    results = []
    for record in data.get("PubTator3", []):
        pmid = record.get("pmid", id)

        # Extract title and abstract
        for passage in record.get("passages", []):
            infons = passage.get("infons", {})
            text = passage.get("text", "")
            if infons.get("type") == "title":
                results.append(f"{pmid} | t | {text}")
            elif infons.get("type") == "abstract":
                results.append(f"{pmid} | a | {text}")

            # Extract annotations
            for annotation in passage.get("annotations", []):
                offset = annotation.get("locations", [{}])[0].get("offset", "")
                length = annotation.get("locations", [{}])[0].get("length", "")
                name = annotation.get("text", "")
                biotype = annotation.get("infons", {}).get("biotype", "")
                if offset and length and name and biotype:
                    results.append(f"{pmid} | {offset} | {length} | {name} | {biotype}")
    
    return results

# Read IDs from data
df_id = read_data(input_data)
id_list = df_id['ID'].tolist()

# Process each ID and write to file immediately
with open(output_data, 'w', encoding='utf-8') as f:  # Specify encoding to handle different character sets
    for pub_id in id_list:
        response = call_api(pub_id)
        if response:
            extracted_data = process_response(response, pub_id)
            for item in extracted_data:
                print(item)  # Print the extracted data
                f.write(item + "\n")
            f.write("\n")  # Add a blank line between records
