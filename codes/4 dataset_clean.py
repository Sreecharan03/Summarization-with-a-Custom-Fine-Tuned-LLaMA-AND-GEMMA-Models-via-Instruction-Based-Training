'''import json
import pandas as pd

# ✅ Define dataset path
dataset_path = "C:\\Users\\ASUS\\Downloads\\output_cleaned.jsonl"

# ✅ Read and parse the JSONL file
data = []
with open(dataset_path, "r", encoding="utf-8") as file:
    for line in file:
        try:
            entry = json.loads(line.strip())  # Parse each line as JSON
            data.append(entry)
        except json.JSONDecodeError:
            print(f"❌ Skipping malformed JSON line: {line[:100]}")  # Show part of the problematic line

# ✅ Convert to DataFrame
df = pd.DataFrame(data)

# ✅ Ensure all required columns exist
required_columns = ["instruction", "context", "response", "category"]
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    print(f"❌ Missing columns: {missing_columns}")
else:
    print("✅ All required columns are present!")

# ✅ Convert 'response' column to string and fill missing values
df["response"] = df["response"].astype(str).fillna("")

# ✅ Define hallucination keywords
hallucination_keywords = ["Supreme Court held", "Section 403", "High Court ruling", "penalty under section", "invalid clause"]

# ✅ Detect potential hallucinations
df["hallucination_detected"] = df["response"].apply(lambda x: any(keyword in x for keyword in hallucination_keywords))

# ✅ Display potential hallucinations
hallucinated_rows = df[df["hallucination_detected"]]
if not hallucinated_rows.empty:
    print("⚠️ Potential Hallucinated Summaries Detected:")
    print(hallucinated_rows[["context", "response"]].head(10))  # Show first few hallucinated rows
else:
    print("✅ No hallucinations detected in the first scan!")

# ✅ Save a cleaned version (Removing hallucinated summaries)
cleaned_df = df[~df["hallucination_detected"]].drop(columns=["hallucination_detected"])
cleaned_dataset_path = "output_cleaned_fixed.jsonl"

cleaned_df.to_json(cleaned_dataset_path, orient="records", lines=True, force_ascii=False)
print(f"✅ Cleaned dataset saved as: {cleaned_dataset_path}")'''
import json
import pandas as pd

# Load the cleaned dataset
dataset_path = "output_cleaned_fixed.jsonl"

# Read the JSONL file
data = []
with open(dataset_path, "r", encoding="utf-8") as file:
    for line in file:
        try:
            entry = json.loads(line.strip())
            data.append(entry)
        except json.JSONDecodeError:
            continue  # Skip malformed lines

# Convert to DataFrame
df = pd.DataFrame(data)

# Print the column names
print("📌 Columns in the dataset:", df.columns)
