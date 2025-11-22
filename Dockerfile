# Gunakan Python 3.11 versi ringan (Slim)
FROM python:3.11-slim

# Set folder kerja
WORKDIR /app

# Install library sistem (Ingat: pakai libgl1)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# --- TAMBAHAN BARU: SETTING TIMEOUT ---
# Memperpanjang batas waktu download pip/poetry agar tidak error "ReadTimeout"
# Kita set ke 2000 detik (sekitar 30 menit)
ENV PIP_DEFAULT_TIMEOUT=2000
ENV POETRY_HTTP_TIMEOUT=2000
# --------------------------------------

# Install Poetry
RUN pip install poetry

# Copy file konfigurasi project
COPY pyproject.toml poetry.lock* ./

# Install dependencies
# Tips: Tambahkan '--without dev' agar Jupyter/Pylint tidak ikut diinstall (Hemat kuota & waktu)
RUN poetry config virtualenvs.create false \
    && poetry install --without dev --no-interaction --no-ansi --no-root

# Copy seluruh kode project
COPY . .

# Install project kita sendiri
RUN poetry install --without dev --no-interaction --no-ansi

# Command default
ENTRYPOINT ["bsort"]
CMD ["--help"]