from transformers import pipeline

# sentiment analysis is a task
classifier = pipeline("sentiment-analysis")

res = classifier("hello world, nice to meet you.I have been waiting for you for a long time")

print(res)