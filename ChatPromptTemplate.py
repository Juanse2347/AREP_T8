from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models  import init_chat_model
from dotenv import load_dotenv

load_dotenv()


system_template = "Translate the following from English into {language}"
prompt_template = ChatPromptTemplate.from_messages(    [("system", system_template), ("user", "{text}")])

prompt = prompt_template.invoke({"language": "Italian", "text": "hi!"})
model = init_chat_model("gpt-4o-mini", model_provider="openai")

print(prompt)
print("\n")
print(prompt.to_messages())

response = model.invoke(prompt)
print("\n")
print(response.content)