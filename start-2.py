import ollama

response=ollama.list()

#chat example
response1=ollama.chat(model="mistral",
                       messages=[{
                           "role": "user",
                            "content": " who you are?"
                            }],
                            stream=True)
for chunk in response1:
    print(chunk['message']['content'], end='', flush=True)


#ollama python library's api designed arounf the ollama
    
#genrate example
res=ollama.generate(model="mistral",
                    prompt="describe the sky",
                    )
#show the generated text
#print(ollama.show(res))

#create a new model with modelfile
ollama.create(
    model="ocean_model",
    from_="mistral",
    system="You are very smart assistant who knows everything about oceans, you are very informative and succinct in your answers. You are also very creative and imaginative in your responses.",
    parameters={"temperature": 0.3}
)
response2=ollama.generate(model="ocean_model", prompt="What is the capital of France?")
#print(response2["response"])
ollama.delete(model="ocean_model")