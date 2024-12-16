import csv
"""
    Appending two csv files that have the columns (no checking is done)
    After I make sure the labeled data is good enough, I use this script to append the new data to the main dataset for the next training loop
"""
columns = ['Message','OK','Insult','Violence','Sexual','Hateful','Flirt','Spam','Aggro']

source_csv = './automod-model/partial-train.csv' # from here
dest_csv = './automod-model/train.csv' # to here
with open(source_csv, "r", newline='', encoding='utf-8') as source_file:
    reader = csv.DictReader(source_file)
    with open(dest_csv, "a", newline='', encoding='utf-8') as dest_file:
        writer = csv.DictWriter(dest_file, fieldnames=columns)
        for row in reader:
            writer.writerow(row)

print('Executed')
