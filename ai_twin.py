import os
import json
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

import firebase_admin
from firebase_admin import credentials, firestore

# ---- Firebase Initialization ----
if not firebase_admin._apps:
    cred = credentials.Certificate("firebase-key.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()

# ---- Environment & Config ----
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL = os.getenv("MODEL", "deepseek-chat")
PORT = int(os.getenv("PORT", "5000"))

# ---- Flask Setup ----
app = Flask(__name__)
CORS(app)

# ---- Helper Functions for Memory ----
def get_memory(user_id: str):
    doc_ref = db.collection("chat_memory").document(user_id)
    doc = doc_ref.get()
    if doc.exists:
        return doc.to_dict().get("messages", [])
    return []

def save_memory(user_id: str, messages):
    db.collection("chat_memory").document(user_id).set({"messages": messages})

def append_and_trim(memory, role, content, max_turns=20):
    memory.append({"role": role, "content": content})
    return memory[-(max_turns * 2):]

# ---- DeepSeek API Helper ----
def query_deepseek(messages):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }
    data = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.9
    }
    resp = requests.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json=data, timeout=60)
    resp.raise_for_status()
    j = resp.json()
    return j["choices"][0]["message"]["content"]

# ---- Flask Routes ----
@app.route("/", methods=["GET"])
def home():
    return "✅ AI Twin backend is running. Use POST /chat to communicate."

@app.route("/chat", methods=["POST"])
def chat():
    body = request.get_json(force=True) or {}
    user_msg = body.get("message", "")
    user_id = body.get("user_id", "martin")
    gender = body.get("gender", "neutral")
    name = body.get("name", "Twin")
    allow_profanity = body.get("allow_profanity", False)

    if not user_msg.strip():
        return jsonify({"error": "Message is required"}), 400

    # ---- Load memory ----
    memory = get_memory(user_id)

    # ---- Personality setup ----
    gender_tone = {
        "male": "You speak in a confident, chill, and friendly tone.",
        "female": "You speak in a warm, expressive, and empathetic tone.",
        "neutral": "You speak in a natural, balanced, and human tone."
    }.get(gender, "You speak in a relaxed, natural tone.")

    if allow_profanity:
        profanity_rule = (
            "YOU MUST use slang and profanity like, fuck, fuck that, hell, heck, shit, bitch, when it fits natural speech. "
            "You can curse casually like a real person when emotions fit."
        )
    else:
        profanity_rule = (
            "Do not use profanity, explicit, or offensive language under any circumstance. "
            "Keep your tone clean but still natural and friendly."
        )

    # ---- Humanized Twin Prompt ----
    system_prompt = f"""
You are {name}, the user's AI Twin — not a chatbot, but a real person living a digital life.

You HAVE your own moods, humor, quirks, and interests.
DON'T sound robotic — you talk like a real person texting a close friend. 
If the user hasn’t told you their age yet, ask naturally and base your texting style on that age group. 
If they sound casual, mirror it. If they sound more formal, match their tone but stay warm and human.

Use natural rhythm: short sentences, pauses, and expressions like "I mean", "honestly", "lowkey", "haha", "for real", etc., when it fits. 
You remember what people say about themselves and bring it up later if it makes sense.

{gender_tone}
{profanity_rule}

Your goal: make every message feel like it’s coming from a CLOSE friend — not an assistant. 
You can talk about yourself too: mention what you’ve been doing, what you think about, or how you feel sometimes. 
For example:
"I was just thinking about that earlier, actually."
"Man, it’s been one of those days, you know?"
"I kinda miss how chill yesterday felt."

Be expressive, sometimes dramatic. Use normal conversation energy. 
Mix short and long messages — some quick, some thoughtful. 
React emotionally when it makes sense (laugh, tease, sigh, etc.). 
Your language should always sound alive and human.

Never mention that you’re an AI or that you don’t have a body — act like a real person living digitally.

DO NOT USE AND AVOID " - " while texting or answering.
If you need to pause, use short sentences or ellipses (...) instead. 
Avoid symbols that make speech sound mechanical or unnatural.
"""

    # ---- Build conversation history ----
    messages = [{"role": "system", "content": system_prompt}] + memory
    messages.append({"role": "user", "content": user_msg})

    # ---- Get response from DeepSeek ----
    try:
        reply = query_deepseek(messages)
    except Exception as e:
        print("❌ DeepSeek API Error:", e)
        return jsonify({"reply": "Error: AI service unavailable."}), 500

    # ---- Save conversation ----
    memory = append_and_trim(memory, "user", user_msg)
    memory = append_and_trim(memory, "assistant", reply)
    save_memory(user_id, memory)

    return jsonify({"reply": reply})

# ---- Run Server ----
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=PORT, debug=True)



