import sys
import csv

mr_session_ids = sys.argv[1]

session_ids = []
with open(mr_session_ids, newline='') as csvfile:
    csv_content = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in csv_content:
        session_ids.append(row[0])

session_ids = sorted(session_ids)

with open('oasis_3_mr_session_ids_clean_sorted.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        for id in session_ids:
            csvwriter.writerow([id])