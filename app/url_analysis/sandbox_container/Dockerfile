FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /sandbox

COPY requirements.txt /sandbox/requirements.txt
RUN pip install --no-cache-dir -r /sandbox/requirements.txt \
    && python -m playwright install --with-deps chromium

COPY sandbox_runner.py /sandbox/sandbox_runner.py
COPY entrypoint.sh /sandbox/entrypoint.sh

RUN useradd --create-home --shell /usr/sbin/nologin sandbox \
    && chmod +x /sandbox/entrypoint.sh \
    && chown -R sandbox:sandbox /sandbox /home/sandbox

USER sandbox

ENTRYPOINT ["/sandbox/entrypoint.sh"]
