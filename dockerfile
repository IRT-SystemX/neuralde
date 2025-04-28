FROM python:3.9-slim
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# Copier le fichier local dans le conteneur
WORKDIR /NEURAL_LIB
COPY neural_de ./neural_de 
COPY pyproject.toml .
COPY README.md .
COPY requirements_python39.txt .
RUN python -m pip install .

ADD neural_de/main.py .

ENTRYPOINT python main.py \
    --input_source_path "/tmp/in"/"$INPUT_SOURCE_PATH" \
    --output_target_path "/tmp/out"/"$OUTPUT_TARGET_PATH" \
    --output_prefix "$OUTPUT_PREFIX" \
    --pipeline_file_path "/tmp"/"$PIPELINE_FILE_PATH"

