import pandas as pd
import random

random.seed(42)

students = []

for i in range(100):

    study_hours = random.randint(1, 10)
    attendance = random.randint(50, 100)
    previous_marks = random.randint(35, 95)
    assignment_score = random.randint(40, 100)
    sleep_hours = random.randint(4, 9)

    # Calculate a simple performance score
    score = (
        study_hours * 5
        + attendance * 0.25
        + previous_marks * 0.30
        + assignment_score * 0.20
        + sleep_hours * 2
    )

    if score >= 85:
        result = "Pass"
    else:
        result = "Fail"

    students.append([
        study_hours,
        attendance,
        previous_marks,
        assignment_score,
        sleep_hours,
        result
    ])


# Create DataFrame

data = pd.DataFrame(
    students,
    columns=[
        "study_hours",
        "attendance",
        "previous_marks",
        "assignment_score",
        "sleep_hours",
        "result"
    ]
)


# Save dataset

data.to_csv(
    "dataset/student_data.csv",
    index=False
)

print("================================")
print("Dataset created successfully!")
print("================================")

print("\nNumber of students:", len(data))

print("\nResult distribution:")
print(data["result"].value_counts())

print("\nFirst 10 students:")
print(data.head(10))

print("\nDataset saved to:")
print("dataset/student_data.csv")