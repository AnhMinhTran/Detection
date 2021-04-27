import csv

with open('file_index.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["ID", "Name"])
    writer.writerow([0, "5 arms - white"])
    writer.writerow([1, "Brittslestars-obsecured"])
    writer.writerow([2, "Brittslestars-other"])
    writer.writerow([3, "Brittslestars-SF"])
    writer.writerow([4, "Prawn"])
    writer.writerow([5, "Seaspider"])
    writer.writerow([6, "Worm"])
