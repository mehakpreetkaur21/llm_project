from tavily import TavilyClient
import gradio as gr
from langchain_ollama import ChatOllama
llm=ChatOllama(model="llama3.2:1b")
tavily_client=TavilyClient(api_key="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")
response_text = "An unexpected error occurred. Please try again later."
def ai_chat(query,chat_history):
    if not query.strip():
        chat_history.append({"role":"assistant","content":"Please enter a valid question."})
        return "",chat_history
    try:
        if "real-time" in query.lower() or "current" or "latest" in query.lower():
            try:
                tavily_response=tavily_client.qna_search(query=query)
                # respons_text=tavily_response.get("answer" , "tavily could'nt fetch the data.")
                response_text=tavily_response
            except Exception as tavily_error:
                response_text=f"Tavily error:{tavily_error}"
        else:
            response=llm.invoke(query)
            response_text=response.content if hasattr(response,"content") else str(response)
        print(f"Response: {response_text}")
        
        if not response_text.strip():
            response_text="I'm sorry. I am not able to unserstand the question. Could you please try again"
        
        chat_history.append({"role":"user","content":query})
        chat_history.append({"role":"assistant","content":response_text})
        return "",chat_history
    except Exception as e:
        print(f"Error : {e}")
        chat_history.append({"role":"assistant","content":"An error occured.Please try again"})
        return "",chat_history
def clear_chat():
    return []
print(dir(tavily_client))
# Function to set up the chatbot interface
def chatbot_interface():
    with gr.Tab("Chat with Chatbot"):
        gr.Markdown("## AI Chat Interface\nChat with the AI model and get your questions answered.")
        chatbox = gr.Chatbot(label="Chat History", type="messages")
        query_input = gr.Textbox(show_label=False, placeholder="Type your question here...", lines=1)
        with gr.Row():
            send_button = gr.Button("Submit")
            clear_button = gr.Button("Clear Chat")
        send_button.click(ai_chat, inputs=[query_input, chatbox], outputs=[query_input, chatbox])
        query_input.submit(ai_chat, inputs=[query_input, chatbox], outputs=[query_input, chatbox])
        clear_button.click(clear_chat, outputs=chatbox)