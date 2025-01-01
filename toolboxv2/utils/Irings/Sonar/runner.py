import os
from datetime import datetime, timedelta
from typing import Optional, Dict
import jwt
from jwt.exceptions import InvalidTokenError
import sqlite3
import uuid
import torch
from bottle import Bottle, request, response

from sonar.inference_pipelines.text import (
    TextToEmbeddingModelPipeline,
    EmbeddingToTextModelPipeline,
    TextToTextModelPipeline
)
from sonar.inference_pipelines.speech import (
    SpeechToEmbeddingModelPipeline,
    SpeechToTextModelPipeline
)
from sonar.models.blaser.loader import load_blaser_model


# Configuration
class Config:
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-this")
    JWT_ALGORITHM = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 30
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
    DB_PATH = "auth.db"


# Database initialization
def init_db():
    conn = sqlite3.connect(Config.DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS api_keys (
            id TEXT PRIMARY KEY,
            key TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_used TIMESTAMP,
            is_active BOOLEAN DEFAULT 1
        )
    ''')
    conn.commit()
    conn.close()


# Model initialization
class ModelManager:
    def __init__(self):
        self.device = Config.DEVICE
        self.dtype = Config.DTYPE
        self._initialize_models()

    def _initialize_models(self):
        # Text pipelines
        self.text_to_embedding = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder",
            tokenizer="text_sonar_basic_encoder",
            device=self.device
        )

        self.embedding_to_text = EmbeddingToTextModelPipeline(
            decoder="text_sonar_basic_decoder",
            tokenizer="text_sonar_basic_encoder",
            device=self.device
        )

        self.text_to_text = TextToTextModelPipeline(
            encoder="text_sonar_basic_encoder",
            decoder="text_sonar_basic_decoder",
            tokenizer="text_sonar_basic_encoder",
            device=self.device
        )

        # Speech pipelines
        self.speech_to_embedding = SpeechToEmbeddingModelPipeline(
            encoder="sonar_speech_encoder_eng",
            device=self.device
        )

        self.speech_to_text = SpeechToTextModelPipeline(
            encoder="sonar_speech_encoder_eng",
            decoder="text_sonar_basic_decoder",
            tokenizer="text_sonar_basic_decoder",
            device=self.device
        )

        # BLASER models
        self.blaser_ref = load_blaser_model("blaser_2_0_ref").to(device=self.device, dtype=self.dtype).eval()
        self.blaser_qe = load_blaser_model("blaser_2_0_qe").to(device=self.device, dtype=self.dtype).eval()


# Initialize application
app = Bottle()
init_db()
model_manager = ModelManager()


# JWT Authentication
class AuthManager:
    @staticmethod
    def create_api_key(name: str) -> Dict[str, str]:
        api_key = str(uuid.uuid4())
        conn = sqlite3.connect(Config.DB_PATH)
        c = conn.cursor()
        c.execute(
            "INSERT INTO api_keys (id, key, name) VALUES (?, ?, ?)",
            (str(uuid.uuid4()), api_key, name)
        )
        conn.commit()
        conn.close()
        return {"api_key": api_key, "name": name}

    @staticmethod
    def revoke_api_key(api_key: str) -> bool:
        conn = sqlite3.connect(Config.DB_PATH)
        c = conn.cursor()
        c.execute("UPDATE api_keys SET is_active = 0 WHERE key = ?", (api_key,))
        success = c.rowcount > 0
        conn.commit()
        conn.close()
        return success

    @staticmethod
    def create_token(api_key: str) -> Optional[str]:
        conn = sqlite3.connect(Config.DB_PATH)
        c = conn.cursor()
        c.execute(
            "SELECT id FROM api_keys WHERE key = ? AND is_active = 1",
            (api_key,)
        )
        result = c.fetchone()

        if result:
            c.execute(
                "UPDATE api_keys SET last_used = CURRENT_TIMESTAMP WHERE key = ?",
                (api_key,)
            )
            conn.commit()

            payload = {
                "sub": result[0],
                "exp": datetime.utcnow() + timedelta(minutes=Config.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
            }
            token = jwt.encode(payload, Config.JWT_SECRET_KEY, algorithm=Config.JWT_ALGORITHM)
            conn.close()
            return token

        conn.close()
        return None

    @staticmethod
    def validate_token(token: str) -> bool:
        try:
            payload = jwt.decode(token, Config.JWT_SECRET_KEY, algorithms=[Config.JWT_ALGORITHM])
            conn = sqlite3.connect(Config.DB_PATH)
            c = conn.cursor()
            c.execute(
                "SELECT 1 FROM api_keys WHERE id = ? AND is_active = 1",
                (payload["sub"],)
            )
            exists = c.fetchone() is not None
            conn.close()
            return exists
        except InvalidTokenError:
            return False


def require_auth(func):
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            response.status = 401
            return {"error": "Missing or invalid authorization header"}

        token = auth_header.split(' ')[1]
        if not AuthManager.validate_token(token):
            response.status = 401
            return {"error": "Invalid or expired token"}

        return func(*args, **kwargs)

    return wrapper


# Auth endpoints
@app.post('/api/keys')
def create_key():
    name = request.json.get('name')
    ak = request.json.get('adKey')
    if ak != os.getenv("TB_R_KEY"):
        return {"error": "Not Public"}
    if not name:
        response.status = 400
        return {"error": "Name is required"}

    return AuthManager.create_api_key(name)


@app.delete('/api/keys/<api_key>')
def revoke_key(api_key):
    if AuthManager.revoke_api_key(api_key):
        return {"message": "API key revoked successfully"}
    response.status = 404
    return {"error": "API key not found"}


@app.post('/api/token')
def get_token():
    api_key = request.json.get('api_key')
    if not api_key:
        response.status = 400
        return {"error": "API key is required"}

    token = AuthManager.create_token(api_key)
    if token:
        return {"token": token}

    response.status = 401
    return {"error": "Invalid API key"}


from typing import Iterator
import json
from io import BytesIO
import numpy as np


def stream_array(arr: np.ndarray) -> Iterator[str]:
    """Stream numpy array as JSON chunks"""
    for row in arr:
        yield json.dumps({'embedding': row.tolist()}) + '\n'


# Add these utility functions after the existing imports

def batch_generator(items, batch_size):
    """Generate batches from a list of items"""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def process_batch_with_progress(items, batch_size, process_fn):
    """Process items in batches and yield progress"""
    total_items = len(items)
    processed = 0

    for batch in batch_generator(items, batch_size):
        result = process_fn(batch)
        processed += len(batch)
        yield {
            'progress': processed / total_items,
            'batch_result': result
        }


# Modify the text_to_embedding endpoint to support batching and streaming
@app.post('/api/embeddings/text')
@require_auth
def text_to_embedding():
    data = request.json
    texts = data.get('texts', [])
    source_lang = data.get('source_lang', 'eng_Latn')
    stream = data.get('stream', False)
    batch_size = data.get('batch_size', 32)

    if not texts:
        response.status = 400
        return {"error": "No texts provided"}

    if stream:
        response.content_type = 'application/x-ndjson'

        def process_batch(batch):
            embeddings = model_manager.text_to_embedding.predict(
                batch,
                source_lang=source_lang
            )
            return embeddings.cpu().numpy()

        return (
            json.dumps(batch_result) + '\n'
            for batch_result in process_batch_with_progress(
            texts,
            batch_size,
            process_batch
        )
        )

    # Non-streaming batch processing
    all_embeddings = []
    for batch in batch_generator(texts, batch_size):
        batch_embeddings = model_manager.text_to_embedding.predict(
            batch,
            source_lang=source_lang
        )
        all_embeddings.append(batch_embeddings)

    combined_embeddings = torch.cat(all_embeddings, dim=0)
    return {"embeddings": combined_embeddings.cpu().numpy().tolist()}


# Similarly modify speech_to_embedding for batching
@app.post('/api/embeddings/speech')
@require_auth
def speech_to_embedding():
    try:
        import torchaudio
    except ImportError:
        response.status = 501
        return {"error": "No torchaudio found"}
    if not request.files:
        response.status = 400
        return {"error": "No audio files provided"}

    stream = request.forms.get('stream', 'false').lower() == 'true'
    batch_size = int(request.forms.get('batch_size', '16'))

    audio_data = []
    for file in request.files.values():
        buf = BytesIO()
        file.save(buf)
        buf.seek(0)

        waveform, sample_rate = torchaudio.load(buf)
        if sample_rate != 16000:
            transform = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = transform(waveform)

        audio_data.append(waveform)

    if stream:
        response.content_type = 'application/x-ndjson'

        def process_batch(batch):
            embeddings = model_manager.speech_to_embedding.predict(batch)
            return embeddings.cpu().numpy()

        return (
            json.dumps(batch_result) + '\n'
            for batch_result in process_batch_with_progress(
            audio_data,
            batch_size,
            process_batch
        )
        )

    # Non-streaming batch processing
    all_embeddings = []
    for batch in batch_generator(audio_data, batch_size):
        batch_embeddings = model_manager.speech_to_embedding.predict(batch)
        all_embeddings.append(batch_embeddings)

    combined_embeddings = torch.cat(all_embeddings, dim=0)
    return {"embeddings": combined_embeddings.cpu().numpy().tolist()}


@app.post('/api/text/from_embeddings')
@require_auth
def embedding_to_text():
    data = request.json
    embeddings = torch.tensor(data.get('embeddings', []), device=model_manager.device)
    target_lang = data.get('target_lang', 'eng_Latn')
    max_seq_len = data.get('max_seq_len', 512)

    if embeddings.nelement() == 0:
        response.status = 400
        return {"error": "No embeddings provided"}

    texts = model_manager.embedding_to_text.predict(
        embeddings,
        target_lang=target_lang,
        max_seq_len=max_seq_len
    )
    return {"texts": texts}


@app.post('/api/translate/text')
@require_auth
def translate_text():
    data = request.json
    texts = data.get('texts', [])
    source_lang = data.get('source_lang', 'eng_Latn')
    target_lang = data.get('target_lang', 'eng_Latn')

    if not texts:
        response.status = 400
        return {"error": "No texts provided"}

    translations = model_manager.text_to_text.predict(
        texts,
        source_lang=source_lang,
        target_lang=target_lang
    )
    return {"translations": translations}


@app.post('/api/speech/to_text')
@require_auth
def speech_to_text():
    try:
        import torchaudio
    except ImportError:
        response.status = 501
        return {"error": "No torchaudio found"}
    if not request.files:
        response.status = 400
        return {"error": "No audio files provided"}

    target_lang = request.forms.get('target_lang', 'eng_Latn')
    audio_data = []

    for file in request.files.values():
        buf = BytesIO()
        file.save(buf)
        buf.seek(0)

        waveform, sample_rate = torchaudio.load(buf)
        if sample_rate != 16000:
            transform = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = transform(waveform)

        audio_data.append(waveform)

    texts = model_manager.speech_to_text.predict(audio_data, target_lang=target_lang)
    return {"texts": texts}


# BLASER Similarity Endpoints
@app.post('/api/similarity/blaser')
@require_auth
def compute_similarity():
    data = request.json
    src_texts = data.get('source_texts', [])
    mt_texts = data.get('translated_texts', [])
    ref_texts = data.get('reference_texts', [])
    source_lang = data.get('source_lang', 'eng_Latn')
    target_lang = data.get('target_lang', 'eng_Latn')

    if not src_texts or not mt_texts:
        response.status = 400
        return {"error": "Source and translated texts are required"}

    with torch.inference_mode():
        src_embs = model_manager.text_to_embedding.predict(src_texts, source_lang=source_lang)
        mt_embs = model_manager.text_to_embedding.predict(mt_texts, source_lang=target_lang)

        if ref_texts:
            ref_embs = model_manager.text_to_embedding.predict(ref_texts, source_lang=target_lang)
            scores = model_manager.blaser_ref(
                src=src_embs.to(dtype=model_manager.dtype),
                ref=ref_embs.to(dtype=model_manager.dtype),
                mt=mt_embs.to(dtype=model_manager.dtype)
            )
        else:
            scores = model_manager.blaser_qe(
                src=src_embs.to(dtype=model_manager.dtype),
                mt=mt_embs.to(dtype=model_manager.dtype)
            )

    return {"scores": scores.cpu().numpy().tolist()}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

