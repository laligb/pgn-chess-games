import requests
import os
import pandas as pd

url = "https://chess-yarc62mhna-ew.a.run.app/upload"

filedir = os.path.join("data", "dataset")
files = os.listdir(filedir)
print(f"Opening {filedir}")

output = open("output_2.csv", "w")
output.write("filename,pgn,missed\n")
print("Writing output.csv")

responses = {}
for file in files:
    filepath = os.path.join(filedir, file)
    if os.path.isfile(filepath) and file.endswith(".png"):
        print(f"Sending {file}...")
        attach = {"img": open(filepath, "rb")}
        response = requests.post(url, files=attach)
        pgn = response.json()["pgn"]
        missed = pgn.count("missed")
        output.write(f"{file},{pgn},{missed}\n")
        print(f"{missed} missed moves.\n")

print("Finished saving output.csv")
print("Loading DataFrame...")

df = pd.read_csv("output.csv", header=0)
df = df.sort_values("missed")

print("Saving ordered_output_2.csv")

df.to_csv("ordered_output_2.csv")

print("Best files to try:")

print(df["filename", "missed"].head(10))
