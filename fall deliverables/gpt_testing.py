from openai import OpenAI
import os

file = open("key.txt", 'r')
key = file.readline()
key = key[:len(key) - 1]


client = OpenAI(api_key=key)


response1 = client.responses.create(
    model = "gpt-5-nano",
    input = "Create a simple math question involving only basic arithmetic"
)

question = response1.output_text


def ask_chat_gpt(prompt, question):
    response = client.chat.completions.create(
        model="gpt-5-nano",

        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question}
        ]
    )

    return response.choices[0].message.content

for i in range(0, 5):
    background = "You are a student taking a test. You will be prompted with simple math questions. Give the correct answer as well as an explanation of what you did to get the answer"
    print(ask_chat_gpt(background, f"({i} / 3 + 17 - 4) ** 2"))

print(ask_chat_gpt(background, question))