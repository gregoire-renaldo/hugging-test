from transformers import pipeline

# sentiment analysis is a task
classifier = pipeline("sentiment-analysis")

res = classifier("hello world, nice to meet you.May I ask you something?")

print(res)