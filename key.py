import google.generativeai as genai

genai.configure(api_key="AIzaSyCYpM6pOM-yWQz4S5gAu8Ch2stogXvg0k8")

models = genai.list_models()

for m in models:
    print(m.name)
