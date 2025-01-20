from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# sentiment analysis is a task
classifier = pipeline("sentiment-analysis")

res = classifier("hello world, nice to meet you.May I ask you something?")

print(res)

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model =  AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

res = classifier("hello world, nice to meet you.May I ask you something?")

print(res)

sequence = "Using transformers is easy!"

res = tokenizer(sequence)
print(res)

tokens = tokenizer.tokenize(sequence)
print(tokens)

ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)


decoded_string = tokenizer.decode(ids)
print(decoded_string)


