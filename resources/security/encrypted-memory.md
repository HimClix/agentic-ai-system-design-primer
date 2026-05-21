# Encrypted Agent Memory

> Encrypting agent memory at rest and in transit: AES-256 for vector stores, field-level encryption for Postgres memory tables, per-tenant key management, and HIPAA/GDPR compliance.

## What It Is

Agent memory contains sensitive data: user PII (names, emails, preferences), conversation history (which may include financial details, health information, or confidential business data), and behavioral patterns. Encrypting this data is not optional for production systems handling real user data.

What needs encryption:
- **Vector store entries**: Embeddings + original text chunks stored in Qdrant/Pinecone/pgvector
- **Conversation history**: Message logs in Postgres/Redis
- **User preferences**: Learned facts about users stored in memory tables
- **Session state**: Workspace/scratchpad data between agent steps

### Threat Model

```
                    ┌─────────────────────────────┐
                    │     ENCRYPTED AT REST         │
                    │                               │
 User ──► Agent ──►│  PostgreSQL (conversations)    │
                    │  Vector DB (embeddings)       │
                    │  Redis (session state)        │
                    │  S3 (exported memories)       │
                    │                               │
                    └─────────────────────────────┘

 Threats:
 1. Database breach → attacker reads all conversations
 2. Backup theft → old conversations exposed
 3. Insider access → DBA reads user conversations
 4. Cross-tenant leakage → tenant A sees tenant B's data
 5. Compliance audit → prove data is encrypted (GDPR Art. 32)
```

## How It Works

### Encryption Architecture

```
┌───────────────────────────────────────────────────────────┐
│                    KEY HIERARCHY                           │
│                                                           │
│  ┌─────────────────┐                                     │
│  │  Master Key (KEK)│  Stored in AWS KMS / HashiCorp Vault│
│  │  (never leaves   │  Rotated annually                   │
│  │   KMS)           │                                     │
│  └────────┬────────┘                                     │
│           │ wraps                                         │
│  ┌────────▼────────┐                                     │
│  │  Tenant Key (DEK)│  One per tenant/customer            │
│  │  (encrypted by   │  Stored encrypted in DB             │
│  │   master key)    │  Rotated quarterly                  │
│  └────────┬────────┘                                     │
│           │ encrypts                                      │
│  ┌────────▼──────────────────────────────────────────┐   │
│  │  Memory Data                                       │   │
│  │  - Conversation messages (AES-256-GCM)            │   │
│  │  - Vector embeddings (AES-256-GCM)                │   │
│  │  - User preferences (field-level AES-256)         │   │
│  │  - Session state (AES-256-GCM)                    │   │
│  └───────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────┘
```

### What Gets Encrypted (and What Does Not)

| Data | Encrypt? | Why |
|------|----------|-----|
| Conversation text | YES | Contains user PII, business data |
| Vector embeddings | DEPENDS | Embeddings can be reversed to approximate original text |
| Embedding metadata (original text) | YES | Contains the actual sensitive content |
| User preferences | YES | PII (name, location, preferences) |
| Agent system prompts | NO | Not user-specific, shared across sessions |
| Tool call logs (names + args) | YES (args only) | Args may contain PII, tool names are not sensitive |
| Session scratchpad | YES | May contain intermediate PII |
| Timestamps, IDs | NO | Non-sensitive metadata |

## Production Implementation

```python
import os
import json
import base64
import hashlib
from dataclasses import dataclass
from typing import Optional
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


@dataclass
class EncryptedPayload:
    """Encrypted data with nonce and metadata."""
    ciphertext: bytes
    nonce: bytes
    key_id: str        # Which key was used (for key rotation)
    version: int = 1   # Encryption scheme version


class MemoryEncryptor:
    """
    AES-256-GCM encryption for agent memory.
    
    Features:
    - Per-tenant encryption keys
    - Key derivation from master secret
    - Nonce management (never reuse)
    - Key rotation support
    - Field-level encryption for database columns
    """

    def __init__(self, master_key: bytes = None):
        """
        Initialize with a master key.
        In production, fetch from AWS KMS or HashiCorp Vault.
        """
        self.master_key = master_key or os.urandom(32)
        self._tenant_keys: dict[str, bytes] = {}

    def _derive_tenant_key(self, tenant_id: str) -> bytes:
        """Derive a unique encryption key per tenant from the master key."""
        if tenant_id in self._tenant_keys:
            return self._tenant_keys[tenant_id]

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=tenant_id.encode(),
            iterations=100_000,
        )
        key = kdf.derive(self.master_key)
        self._tenant_keys[tenant_id] = key
        return key

    def encrypt(self, plaintext: str, tenant_id: str) -> EncryptedPayload:
        """Encrypt a string using AES-256-GCM."""
        key = self._derive_tenant_key(tenant_id)
        nonce = os.urandom(12)  # 96-bit nonce for GCM
        aesgcm = AESGCM(key)

        ciphertext = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)

        return EncryptedPayload(
            ciphertext=ciphertext,
            nonce=nonce,
            key_id=f"tenant:{tenant_id}:v1",
        )

    def decrypt(self, payload: EncryptedPayload, tenant_id: str) -> str:
        """Decrypt an encrypted payload."""
        key = self._derive_tenant_key(tenant_id)
        aesgcm = AESGCM(key)

        plaintext = aesgcm.decrypt(payload.nonce, payload.ciphertext, None)
        return plaintext.decode("utf-8")

    def encrypt_for_storage(self, plaintext: str, tenant_id: str) -> str:
        """Encrypt and encode as base64 string for database storage."""
        payload = self.encrypt(plaintext, tenant_id)
        combined = json.dumps({
            "c": base64.b64encode(payload.ciphertext).decode(),
            "n": base64.b64encode(payload.nonce).decode(),
            "k": payload.key_id,
            "v": payload.version,
        })
        return combined

    def decrypt_from_storage(self, stored: str, tenant_id: str) -> str:
        """Decrypt a base64-encoded stored value."""
        data = json.loads(stored)
        payload = EncryptedPayload(
            ciphertext=base64.b64decode(data["c"]),
            nonce=base64.b64decode(data["n"]),
            key_id=data["k"],
            version=data["v"],
        )
        return self.decrypt(payload, tenant_id)


# --- Field-Level Encryption for PostgreSQL ---

POSTGRES_SCHEMA = """
-- Conversations table with encrypted fields
CREATE TABLE agent_conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(64) NOT NULL,
    session_id VARCHAR(64) NOT NULL,
    
    -- ENCRYPTED FIELDS (stored as base64 JSON)
    user_message_encrypted TEXT NOT NULL,      -- AES-256-GCM encrypted
    assistant_response_encrypted TEXT NOT NULL, -- AES-256-GCM encrypted
    tool_calls_encrypted TEXT,                 -- AES-256-GCM encrypted
    
    -- PLAINTEXT FIELDS (non-sensitive metadata)
    message_index INTEGER NOT NULL,
    model_used VARCHAR(64),
    token_count_input INTEGER,
    token_count_output INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Encryption metadata
    encryption_key_id VARCHAR(128) NOT NULL,
    encryption_version INTEGER DEFAULT 1
);

-- Index on non-encrypted fields only
CREATE INDEX idx_conversations_tenant ON agent_conversations(tenant_id);
CREATE INDEX idx_conversations_session ON agent_conversations(session_id);
CREATE INDEX idx_conversations_created ON agent_conversations(created_at);

-- User memory table with encrypted preferences
CREATE TABLE agent_user_memory (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(64) NOT NULL,
    user_id VARCHAR(64) NOT NULL,
    
    -- ENCRYPTED
    memory_content_encrypted TEXT NOT NULL,    -- The actual memory
    
    -- PLAINTEXT (for querying)
    memory_type VARCHAR(32) NOT NULL,         -- "preference", "fact", "instruction"
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_accessed_at TIMESTAMPTZ DEFAULT NOW(),
    access_count INTEGER DEFAULT 0,
    
    encryption_key_id VARCHAR(128) NOT NULL
);
"""


class EncryptedConversationStore:
    """PostgreSQL store with field-level encryption."""

    def __init__(self, db_pool, encryptor: MemoryEncryptor):
        self.db = db_pool
        self.encryptor = encryptor

    async def save_message(
        self,
        tenant_id: str,
        session_id: str,
        user_message: str,
        assistant_response: str,
        tool_calls: list[dict] = None,
        model: str = "claude-sonnet",
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> str:
        """Save an encrypted conversation turn."""
        encrypted_user = self.encryptor.encrypt_for_storage(user_message, tenant_id)
        encrypted_response = self.encryptor.encrypt_for_storage(assistant_response, tenant_id)
        encrypted_tools = (
            self.encryptor.encrypt_for_storage(json.dumps(tool_calls), tenant_id)
            if tool_calls else None
        )

        row = await self.db.fetchrow(
            """
            INSERT INTO agent_conversations 
                (tenant_id, session_id, user_message_encrypted, 
                 assistant_response_encrypted, tool_calls_encrypted,
                 message_index, model_used, token_count_input, 
                 token_count_output, encryption_key_id)
            VALUES ($1, $2, $3, $4, $5, 
                    (SELECT COALESCE(MAX(message_index), 0) + 1 
                     FROM agent_conversations WHERE session_id = $2),
                    $6, $7, $8, $9)
            RETURNING id
            """,
            tenant_id, session_id, encrypted_user, encrypted_response,
            encrypted_tools, model, input_tokens, output_tokens,
            f"tenant:{tenant_id}:v1",
        )
        return str(row["id"])

    async def load_conversation(
        self, tenant_id: str, session_id: str
    ) -> list[dict]:
        """Load and decrypt a conversation."""
        rows = await self.db.fetch(
            """
            SELECT * FROM agent_conversations 
            WHERE tenant_id = $1 AND session_id = $2
            ORDER BY message_index
            """,
            tenant_id, session_id,
        )

        messages = []
        for row in rows:
            messages.append({
                "user_message": self.encryptor.decrypt_from_storage(
                    row["user_message_encrypted"], tenant_id
                ),
                "assistant_response": self.encryptor.decrypt_from_storage(
                    row["assistant_response_encrypted"], tenant_id
                ),
                "tool_calls": (
                    json.loads(self.encryptor.decrypt_from_storage(
                        row["tool_calls_encrypted"], tenant_id
                    ))
                    if row["tool_calls_encrypted"] else None
                ),
                "model": row["model_used"],
                "created_at": row["created_at"].isoformat(),
            })

        return messages


# --- AWS KMS Integration (Production) ---

KMS_EXAMPLE = """
import boto3

class KMSKeyManager:
    def __init__(self, region: str = "ap-south-1"):
        self.kms = boto3.client("kms", region_name=region)
    
    def create_tenant_key(self, tenant_id: str) -> str:
        response = self.kms.create_key(
            Description=f"Agent memory key for tenant {tenant_id}",
            KeyUsage="ENCRYPT_DECRYPT",
            KeySpec="AES_256",
            Tags=[{"TagKey": "tenant_id", "TagValue": tenant_id}],
        )
        return response["KeyMetadata"]["KeyId"]
    
    def encrypt(self, key_id: str, plaintext: bytes) -> bytes:
        response = self.kms.encrypt(
            KeyId=key_id,
            Plaintext=plaintext,
        )
        return response["CiphertextBlob"]
    
    def decrypt(self, ciphertext: bytes) -> bytes:
        response = self.kms.decrypt(CiphertextBlob=ciphertext)
        return response["Plaintext"]
"""
```

## Decision Tree: What to Encrypt

```
    Should I encrypt this data?
                │
     ┌──────────▼──────────┐
     │ Does it contain      │
     │ user-generated       │
     │ content?             │
     └──┬──────────────┬──┘
       Yes             No
        │               │
    ENCRYPT         ┌───▼───────────────┐
                    │ Is it agent-       │
                    │ internal metadata? │
                    │ (timestamps, IDs,  │
                    │  model names)      │
                    └──┬────────────┬───┘
                      Yes          No
                       │            │
                  DON'T ENCRYPT   Evaluate case-by-case
```

## When NOT to Use

1. **Development/testing environments**: Encryption adds complexity to debugging. Use in production only or with easy toggle.
2. **Embeddings you need to search**: Encrypted embeddings cannot be searched with vector similarity. You must encrypt metadata but leave embeddings searchable (or use encrypted search schemes).
3. **High-throughput, low-sensitivity data**: If the data is public knowledge (not PII), encryption overhead is not justified.
4. **When using managed vector DBs with built-in encryption**: Pinecone, Weaviate Cloud, and Qdrant Cloud encrypt at rest by default. Adding application-level encryption is double-encrypting.
5. **Performance-critical real-time paths**: AES-256-GCM encryption adds ~0.1ms per operation. At 10K ops/second, this is 1 second of added latency.

## Tradeoffs

| Advantage | Disadvantage |
|-----------|--------------|
| GDPR/HIPAA compliance | Cannot search encrypted text fields |
| Per-tenant isolation (key per tenant) | Key management complexity |
| Protection against DB breach | Performance overhead (~0.1ms/encrypt) |
| Audit trail (which key, when) | Debugging encrypted data is harder |
| Data portability (export encrypted) | Key loss = data loss (unrecoverable) |

## GDPR / HIPAA Implications

| Requirement | How Encryption Helps |
|------------|---------------------|
| GDPR Art. 32 (security of processing) | AES-256 encryption is an "appropriate technical measure" |
| GDPR Art. 17 (right to erasure) | Delete the tenant key = all data unrecoverable (crypto-shredding) |
| HIPAA (PHI protection) | Encryption is a required safeguard for ePHI |
| Data breach notification | Encrypted data breach may not require notification (safe harbor) |

## Failure Modes

### 1. Key Loss
The encryption key is deleted or corrupted, making all encrypted data permanently unrecoverable.
**Mitigation**: Key backups in a separate KMS. Multi-region key replication. Never store keys in the same DB as data.

### 2. Performance Degradation
High-volume agents encrypt/decrypt thousands of messages, adding latency.
**Mitigation**: Encrypt only sensitive fields (not metadata). Use hardware-accelerated AES (AES-NI). Cache decrypted data in memory.

### 3. Nonce Reuse
Reusing a GCM nonce with the same key breaks the encryption entirely.
**Mitigation**: Always use `os.urandom(12)` for nonces. Never derive nonces deterministically.

### 4. Key Rotation Gap
Old data encrypted with old keys becomes inaccessible after rotation.
**Mitigation**: Re-encrypt old data during rotation. Keep old keys accessible (but marked deprecated) during transition.

### 5. Embedding Search Incompatibility
Encrypted embeddings cannot be used for similarity search.
**Mitigation**: Store embeddings unencrypted (they are not directly PII) but encrypt the associated text/metadata.

## Sources and Further Reading

- [GDPR Article 32 - Security of Processing](https://gdpr.eu/article-32-security-of-processing/)
- [HIPAA Security Rule - Encryption](https://www.hhs.gov/hipaa/for-professionals/security/guidance/index.html)
- [AWS KMS Documentation](https://docs.aws.amazon.com/kms/)
- [Python cryptography library](https://cryptography.io/en/latest/)
- [HashiCorp Vault](https://www.vaultproject.io/)
- [Pinecone Security](https://docs.pinecone.io/docs/security) -- Built-in encryption at rest
