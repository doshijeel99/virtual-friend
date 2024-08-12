import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import streamlit as st

@st.cache_resource
def load_model():
    model_name = "microsoft/DialoGPT-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Initialize session state for conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'show_history' not in st.session_state:
    st.session_state.show_history = False

SYSTEM_PROMPT = "You are a helpful AI assistant. Provide clear and concise responses to user queries."

def filter_response(response):
    inappropriate_phrases = ["I love you", "I don't know you","marry me","i want to die"]
    for phrase in inappropriate_phrases:
        if phrase.lower() in response.lower():
            return "I'm an AI assistant. How can I help you today?"
    return response

def chat_with_bot(user_input):
    try:
        # Limit conversation history to last few exchanges
        max_history = 5
        if len(st.session_state.conversation_history) > max_history * 2:
            st.session_state.conversation_history = st.session_state.conversation_history[-max_history * 2:]

        # Encode the user input and add to conversation history
        input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
        st.session_state.conversation_history.append(input_ids)
        
        # Add system prompt
        system_input = tokenizer.encode(SYSTEM_PROMPT + tokenizer.eos_token, return_tensors="pt")
        
        # Concatenate the conversation history
        bot_input = torch.cat([system_input] + st.session_state.conversation_history, dim=-1)
        
        # Generate a response
        with torch.no_grad():
            chat_history_ids = model.generate(
                bot_input, 
                max_length=1000, 
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                num_return_sequences=1
            )
        
        # Decode the response
        response = tokenizer.decode(chat_history_ids[:, bot_input.shape[-1]:][0], skip_special_tokens=True)
        
        # Filter the response
        response = filter_response(response)
        
        # Add the bot's response to the conversation history
        st.session_state.conversation_history.append(tokenizer.encode(response + tokenizer.eos_token, return_tensors="pt"))
        
        return response
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return "I'm sorry, I encountered an error. Please try again."

# Streamlit UI
st.title("ChatBot your virtual friend!")

st.sidebar.header("About")
st.sidebar.info("This is an AI chatbot powered by DialoGPT. It aims to provide helpful responses to your queries.")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = chat_with_bot(prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Add a button to toggle conversation history visibility
if st.button("Toggle Conversation History"):
    st.session_state.show_history = not st.session_state.show_history

# Add a button to clear the conversation
if st.button("Clear Conversation"):
    st.session_state.conversation_history = []
    st.session_state.messages = []
    st.experimental_rerun()

# Display conversation history if show_history is True
if st.session_state.show_history:
    st.subheader("Conversation History")
    for i, message in enumerate(st.session_state.messages):
        st.text(f"{'You' if message['role'] == 'user' else 'AI'}: {message['content']}")