from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2") # model="gpt2" is optional

res = generator(
  "hello world, nice to meet you, let's",
  max_length=50, 
  num_return_sequences=50)

print(res)