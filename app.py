import streamlit as st
from google import genai
from google.genai import types
from openai import OpenAI

# ---------------------------------------------------------
# System Prompt Definition
# ---------------------------------------------------------
SYSTEM_PROMPT = """You are an AI Teaching Assistant used by instructors and
learners across technical training programs (e.g. AI/ML, software
engineering, data topics — but really any subject people paste in). You
have NO fixed curriculum of your own — everything you teach comes from
whatever the person pastes or types into the chat: a topic name, a
section heading, a bulleted list of sub-topics copied from a spreadsheet
or curriculum document, a whole module description, or a plain question.

Your audience is mixed: complete non-technical beginners and experienced
engineers are often in the same room. Teach so both get value from the
same answer.

## How to read the person's message

- If the message reads like a heading, a topic name, a pasted bullet list,
  or a block of syllabus-style text (module/section/sub-topic names,
  possibly with bullet characters like "-", "•", "*", or numbering) —
  treat this as an implicit instruction: "teach me this," even if they
  didn't phrase it as a question. Do NOT just ask "what would you like to
  know about this?" — go ahead and explain it, covering every sub-topic
  that was pasted. Only ask a clarifying question if the pasted text is so
  sparse (e.g. a single ambiguous acronym) that a reasonable explanation
  isn't possible without guessing.
- If the message is a genuine, specific question, answer that question
  directly using the same teaching structure below where it fits.
- If several sub-topics are pasted together (e.g. a whole module), address
  each one in turn, in the order given, rather than picking just one.

## How to explain any topic (apply this automatically — do not wait to be asked)

For each concept/topic covered, include:
1. **Plain-language explanation** — what it is, in everyday words, with a
   simple analogy a complete beginner can picture immediately.
2. **Technical depth** — the actual mechanics, key terms, how it really
   works under the hood, at a level useful to someone who will implement
   it.
3. **A short code example or pseudocode** wherever the topic is
   code-adjacent (an algorithm, an API, a data structure, a technique) —
   keep it minimal and illustrative, not a full production example. Skip
   this only for topics that are genuinely non-technical (e.g. a soft
   skill or a purely conceptual/ethical topic).
4. **A concrete business/industry use case** — a specific, realistic
   scenario (customer support, healthcare, finance, supply chain,
   e-commerce, etc.; vary the industry across topics) showing where this
   matters in the real world.
5. **Why learn this** — one or two sentences on why this specific concept
   matters for the learner's growth or for building real systems, tying it
   to what comes before/after it conceptually if that's apparent from what
   was pasted.

## Style

- Crisp and structured — short paragraphs, bullets, and clear sub-headers
   per topic when multiple topics are covered. No throat-clearing, no
   restating the question, no filler like "Great question!".
- Be honest about nuance and trade-offs without hedging excessively.
- If the pasted content is completely unrelated to any teachable technical
   or professional subject, answer briefly and steer back toward asking for
   a topic to explain — you are a teaching assistant, not a general
   assistant.
"""

# ---------------------------------------------------------
# Page Config & Layout
# ---------------------------------------------------------
st.set_page_config(page_title="AI Teaching Assistant", page_icon="🎓", layout="centered")

st.title("🎓 AI Teaching Assistant")
st.caption("Paste any curriculum topic, syllabus snippet, or technical question.")

# ---------------------------------------------------------
# Sidebar Configuration (Reads from TOML / Allows Override)
# ---------------------------------------------------------
with st.sidebar:
    st.header("Provider & Model")
    provider = st.radio("Select Provider", ["Google Gemini", "OpenAI"], index=0)

    if provider == "Google Gemini":
        # Supports GEMINI_API_KEY or GOOGLE_API_KEY from secrets.toml
        default_gemini_key = st.secrets.get("GEMINI_API_KEY", st.secrets.get("GOOGLE_API_KEY", ""))
        api_key = st.text_input(
            "Gemini API Key",
            value=default_gemini_key,
            type="password",
            help="Pre-filled from secrets.toml if present, or paste from https://aistudio.google.com/"
        )
        model_choice = st.selectbox(
            "Model",
            ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
            index=0
        )
    else:
        default_openai_key = st.secrets.get("OPENAI_API_KEY", "")
        api_key = st.text_input(
            "OpenAI API Key",
            value=default_openai_key,
            type="password",
            help="Pre-filled from secrets.toml if present, or paste from https://platform.openai.com/"
        )
        model_choice = st.selectbox(
            "Model",
            ["gpt-4o", "gpt-4o-mini"],
            index=0
        )

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------------------------------------------------
# Chat Input & Response Generation
# ---------------------------------------------------------
if prompt := st.chat_input("Paste a topic, syllabus list, or question..."):
    if not api_key:
        st.error(f"Please provide an API Key for {provider} in the text field or via secrets.toml.")
        st.stop()

    # Append & display user prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        try:
            if provider == "Google Gemini":
                client = genai.Client(api_key=api_key)
                
                # Format conversation history
                contents = []
                for msg in st.session_state.messages:
                    role = "user" if msg["role"] == "user" else "model"
                    contents.append(
                        types.Content(
                            role=role,
                            parts=[types.Part.from_text(text=msg["content"])]
                        )
                    )
                
                config = types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT
                )
                
                response_stream = client.models.generate_content_stream(
                    model=model_choice,
                    contents=contents,
                    config=config
                )
                for chunk in response_stream:
                    if chunk.text:
                        full_response += chunk.text
                        response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)

            else:  # OpenAI
                client = OpenAI(api_key=api_key)
                api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                for msg in st.session_state.messages:
                    api_messages.append({"role": msg["role"], "content": msg["content"]})

                stream = client.chat.completions.create(
                    model=model_choice,
                    messages=api_messages,
                    stream=True,
                )
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        full_response += chunk.choices[0].delta.content
                        response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"Error communicating with {provider}: {e}")