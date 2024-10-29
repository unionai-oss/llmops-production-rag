"""Utility functions."""

import base64
import io
import os

from functools import wraps, partial
from flytekit import current_context


def env_secret(fn=None, *, secret_name: str, env_var: str):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        os.environ[env_var] = current_context().secrets.get(key=secret_name)
        return fn(*args, **kwargs)

    if fn is None:
        return partial(env_secret, secret_name=secret_name, env_var=env_var)

    return wrapper


openai_env_secret = partial(
    env_secret,
    secret_name="openai_api_key",
    env_var="OPENAI_API_KEY",
)


def convert_fig_into_html(fig) -> str:
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format="png")
    img_base64 = base64.b64encode(img_buf.getvalue()).decode()
    return f'<img src="data:image/png;base64,{img_base64}" alt="Rendered Image" />'
