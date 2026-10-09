import json

# paste your grade function here (copy it from ex1.py, WITHOUT the three print lines at the bottom)
def grade(marks):
    average = sum(marks)/len(marks)
    if average >= 75:
        letter = "A"
    elif average >= 60:
        letter = "B"
    elif average >= 40:
        letter = "C"
    else:
        letter = "F"
    return average, letter

with open("students.json") as f:
    students = json.load(f)

for s in students:
    average, letter = grade(s["marks"])
    print(s["name"], average, letter)