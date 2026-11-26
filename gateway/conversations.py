import uuid

from gateway.memory import memory_store

MAX_HISTORY_MESSAGES = 20

def create_conversation_id() -> str:
    return str(uuid.uuid4())

def get_or_create_conversation(conversation_id: str):
     return memory_store.get_or_create_conversation(conversation_id)

def add_messages(
        conversation_id: str,
        messages: list,
):
   memory_store.add_messages(conversation_id, messages)

def add_assistant_message(
        conversation_id: str,
        content: str,
):
        add_messages(
             conversation_id,
             [
                  {
                       "role": "assistant",
                       "content": content,
                  }
             ],
        )

def get_messages(conversation_id: str):
    return memory_store.get_messages(conversation_id)

def build_conversation_messages(conversation_id: str, new_messages: list):
    history = get_messages(conversation_id)
    return (history + new_messages)[-MAX_HISTORY_MESSAGES:]

def list_conversations():
     return memory_store.list_conversations()

def get_conversation(conversation_id: str):
     return memory_store.get_conversation(conversation_id)

def delete_conversation(conversation_id: str):
     return memory_store.delete_conversation(conversation_id)