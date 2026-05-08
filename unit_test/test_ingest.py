import os
from pathlib import Path
from agentic_ai_platform.rag.vector_rag.ingest import load_vector_store

file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")

if not Path(os.path.join(file_path, "Resume.pdf")).is_file():
    raise FileNotFoundError(f"File not found: {file_path}/Resume.pdf")

load_vector_store(directory_path=file_path, file_type="pdf")