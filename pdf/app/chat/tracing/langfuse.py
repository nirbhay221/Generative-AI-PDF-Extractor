import os
from langfuse import Langfuse

_public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
_secret_key = os.environ.get("LANGFUSE_SECRET_KEY")

langfuse = Langfuse(public_key=_public_key, secret_key=_secret_key) if (_public_key and _secret_key) else None
