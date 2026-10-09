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

print(grade([80, 70, 90]))
print(grade([50, 45, 40]))
print(grade([30, 20, 10]))