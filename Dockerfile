FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml requirements.txt README.md LICENSE ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY data ./data
COPY tests ./tests
COPY run_all.sh ./
RUN pip install --no-cache-dir -e .[test]
CMD ["bash", "run_all.sh"]

