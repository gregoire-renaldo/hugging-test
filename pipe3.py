from transformers import pipeline

classifier = pipeline("zero-shot-classification")

res = classifier(
  "this is about Python list comprehension",
  candidate_labels = ["education", "politics", "business", "sport" ]
)

print(res)