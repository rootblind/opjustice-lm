import csv
import regex as re

# curating for the unlabeled dataset

alphabet_pattern = re.compile(r'^[a-zA-Z]')
url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

data = []

def filter(message):
    if len(message) < 3:
        return message
    message = message.replace('+rep', '')
    message = message.replace('-rep', '')
    message = message.replace('\n', ' ').replace('\r', ' ')
    message = message.lstrip()

    return message


with open('./automod-model/sample.csv', 'r', newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    with open('./automod-model/train-unlabeled.csv', 'w', newline='', encoding='utf-8') as fw:
        writer = csv.DictWriter(fw, fieldnames=['Message'])
        writer.writeheader()
        for row in reader:
            message = row['Message']
            message = filter(message)
            if len(message) > 2 and alphabet_pattern.search(message) and not url_pattern.fullmatch(message):
                data.append(message)
        data = list(set(data))
        for d in data:
            writer.writerow({'Message': d})