import regex as re
import csv
import pandas as pd

def filter_dataset(source):
    # source is the list of rows
    alphabet_pattern = re.compile(r'^[a-zA-Z]')
    allowed_pattern = re.compile(r'[^a-zA-Z0-9 ]')
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

    messages = []
    def filter(message):
        if len(message) < 3:
            return message
        message = message.replace('+rep', '')
        message = message.replace('-rep', '')
        message = message.replace('\n', ' ').replace('\r', ' ')
        message = re.sub(url_pattern, '', message)
        message = re.sub(allowed_pattern, '', message)
        message = message.lstrip()
        return message
    
    for row in source:
        message = row['Message']
        message = filter(message)
        if ' ' not in message:
            continue
        if len(message) > 2 and alphabet_pattern.search(message):
            row['Message'] = message
            messages.append(row)
    print('Filter executed.')

    return messages

def remove_duplicates(data):
    pd_data = {
        "Message": [],
        "OK": [],
        "Aggro": [],
        "Violence": [],
        "Sexual": [],
        "Hateful": []
    }
    columns = ["Message", "OK", "Aggro", "Violence", "Sexual", "Hateful"]
    # converting the data to pd dataframe format to remove duplicates
    for row in data:
        for label in columns:
            pd_data[label].append(row[label])
    df = pd.DataFrame(data=pd_data)
    df.drop_duplicates(subset="Message", inplace=True)
    
    return df



def load_source(filename):
    data = []
    with open(filename, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data
def write_csv(source, destination):
    # source is the list of rows
    columns = ['Message','OK','Insult','Violence','Sexual','Hateful','Flirt','Spam','Aggro']
    with open(destination, "w", newline='', encoding='utf-8') as dest_file:
        writer = csv.DictWriter(dest_file, fieldnames=columns)
        writer.writeheader()
        for row in source:
            writer.writerow(row)

    print(f'{len(source)} rows written.')

reader = load_source('./automod-model/train.csv')
data = filter_dataset(reader)
df_unique = remove_duplicates(data)
df_unique.to_csv('./automod-model/curated.csv', index=False, encoding='utf-8')
print("Done")