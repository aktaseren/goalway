from abc import ABC, abstractmethod
import json
import os
import redis
import psycopg2

from gateway.memory_config import(
    MEMORY_BACKEND,
    REDIS_HOST,
    REDIS_PORT,
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_DB,
    POSTGRES_USER,
    POSTGRES_PASSWORD,
)

class MemoryStore(ABC):
    @abstractmethod
    def get_or_create_conversation(self, conversation_id: str) -> list:
        pass

    @abstractmethod
    def add_messages(self, conversation_id: str, messages: list):
        pass

    @abstractmethod
    def get_messages(self, conversation_id: str) -> list:
        pass

    @abstractmethod
    def list_conversations(self) -> list:
        pass

    @abstractmethod
    def get_conversation(self, conversation_id: str):
        pass

    @abstractmethod
    def delete_conversation(self, conversation_id: str):
        pass

class InMemoryStore(MemoryStore):
    def __init__(self):
        self.conversations = {}
    
    def get_or_create_conversation(self, conversation_id: str) -> list:
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        return self.conversations[conversation_id]
    
    def add_messages(self, conversation_id: str, messages: list):
        conversation = self.get_or_create_conversation(conversation_id)
        conversation.extend(messages)

    def get_messages(self, conversation_id: str) -> list:
        return self.get_or_create_conversation(conversation_id)
    
    def list_conversations(self) -> list:
        return list(self.conversations.keys())
    
    def get_conversation(self, conversation_id: str):
        return self.conversations.get(conversation_id)
    
    def delete_conversation(self, conversation_id: str):
        return self.conversations.pop(conversation_id, None)
    
class RedisStore(MemoryStore):
    def __init__(self):
        self.client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=True,
        )
    
    def _key(self, conversation_id: str):
        return f"conversation: {conversation_id}"
    
    def get_or_create_conversation(self, conversation_id: str) -> list:
        messages = self.get_conversation(conversation_id)
        if messages is None:
            self.client.set(self._key(conversation_id), json.dumps([]))
            return []
        return messages

    def add_messages(self, conversation_id: str, messages: list):
        conversation = self.get_or_create_conversation(conversation_id)
        conversation.extend(messages)
        self.client.set(self._key(conversation_id), json.dumps(conversation))

    def get_messages(self, conversation_id: str) -> list:
        return self.get_or_create_conversation(conversation_id)
    
    def list_conversations(self) -> list:
        keys = self.client.keys("conversation:*")
        return [key.replace("conversation:", "") for key in keys]
    
    def get_conversation(self, conversation_id: str):
        value = self.client.get(self._key(conversation_id))
        if value is None:
            return None
        return json.loads(value)
    
    def delete_conversation(self, conversation_id: str):
        conversation = self.get_conversation(conversation_id)
        if conversation is not None:
            self.client.delete(self._key(conversation_id))
        return conversation
    
class PostgresStore(MemoryStore):
    def __init__(self):
        self.conn = psycopg2.connect(
            host = POSTGRES_HOST,
            port = POSTGRES_PORT,
            dbname = POSTGRES_DB,
            user = POSTGRES_USER,
            password = POSTGRES_PASSWORD,
        )
        self._init_table()
    
    def _init_table(self):
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                        conversation_id TEXT PRIMARY KEY,
                        messages JSONB NOT NULL
                        )
            """)
            self.conn.commit()

    def get_or_create_conversation(self, conversation_id: str) -> list:
        conversation = self.get_conversation(conversation_id)
        if conversation is None:
            self.add_messages(conversation_id, [])
            return []
        return conversation
    
    def add_messages(self, conversation_id: str, messages: list):
        conversation = self.get_conversation(conversation_id) or []
        conversation.extend(messages)

        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO conversations (conversation_id, messages)
                VALUES (%s, %s)
                ON CONFLICT (conversation_id)
                DO UPDATE SET messages = EXCLUDED.messages
                """,
                (conversation_id, json.dumps(conversation)),
            )
            self.conn.commit()
    
    def get_messages(self, conversation_id: str) -> list:
        return self.get_or_create_conversation(conversation_id)
    
    def list_conversations(self) -> list:
        with self.conn.cursor() as cur:
            cur.execute("SELECT conversation_id FROM conversations")
            return [row[0] for row in cur.fetchall()]

    def get_conversation(self, conversation_id: str):
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT messages FROM conversations WHERE conversation_id = %s",
                (conversation_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return row[0]
    
    def delete_conversation(self, conversation_id: str):
        conversation = self.get_conversation(conversation_id)

        with self.conn.cursor() as cur:
            cur.execute(
                "DELETE FROM conversations WHERE conversation_id = %s",
                (conversation_id,),
            )
            self.conn.commit()
        return conversation
    
class CompositeMemoryStore(MemoryStore):
    def __init__(self, primary: MemoryStore, secondary: MemoryStore):
        self.primary = primary
        self.secondary = secondary
    def get_or_create_conversation(self, conversation_id: str) -> list:
        conversation = self.primary.get_conversation(conversation_id)
        if conversation is None:
            conversation = self.secondary.get_conversation(conversation_id)
            if conversation is not None:
                self.primary.add_messages(conversation_id, conversation)
                return conversation
            self.primary.add_messages(conversation_id, [])
            self.secondary.add_messages(conversation_id, [])
            return []
        return conversation
    def add_messages(self, conversation_id: str, messages: list):
        self.primary.add_messages(conversation_id, messages)
        self.secondary.add_messages(conversation_id, messages)
    def get_messages(self, conversation_id: str) -> list:
        return self.get_or_create_conversation(conversation_id)
    def list_conversations(self) -> list:
        return self.secondary.list_conversations()
    def get_conversation(self, conversation_id: str):
        return self.get_or_create_conversation(conversation_id)
    def delete_conversation(self, conversation_id: str):
        primary_deleted = self.primary.delete_conversation(conversation_id)
        secondary_deleted = self.secondary.delete_conversation(conversation_id)
        return secondary_deleted or primary_deleted
                    
MEMORY_BACKEND = MEMORY_BACKEND

if MEMORY_BACKEND == "redis":
    memory_store = RedisStore()
elif MEMORY_BACKEND == "postgres":
    memory_store = PostgresStore()
elif MEMORY_BACKEND == "composite":
    memory_store = CompositeMemoryStore(
        primary = RedisStore(),
        secondary = PostgresStore(),
    )
else:   
    memory_store = InMemoryStore()