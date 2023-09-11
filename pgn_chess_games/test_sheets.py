import requests
import os
import pandas as pd

url = "http://127.0.0.1:8000/upload"

filedir = os.path.join("data", "dataset")
files = os.listdir(filedir)
print(f"Opening {filedir}")

output = open("output_3.csv", "w")
output.write("filename,pgn,missed\n")
print("Writing output_3.csv")

responses = {}
for file in files:
    filepath = os.path.join(filedir, file)
    if os.path.isfile(filepath) and file.endswith(".png"):
        print(f"Sending {file}...")
        attach = {"img": open(filepath, "rb")}
        response = requests.post(url, files=attach)
        pgn = response.json()["pgn"]
        missed = pgn.count("[")
        output.write(f"{file},{pgn},{missed}\n")
        print(f"{missed} missed moves.\n")

print("Finished saving output.csv")
print("Loading DataFrame...")

df = pd.read_csv("output.csv", header=0)
df = df.sort_values("missed", ignore_index=True)

print("Saving ordered_output_3.csv")

df.to_csv("ordered_output_3.csv")

print("Best files to try:")

print(df[["filename", "missed"]].head(10))
