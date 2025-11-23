# Fractal Curve Tokenizer (Podman Edition)

This is a basic setup for a Python project running with Podman.

## Prerequisites

- Podman
- Podman Compose (optional, but recommended)

## Running the project

1.  Build and run with Podman Compose:

    ```bash
    podman-compose up --build
    ```

    Or if you have `podman-docker` emulation or newer podman versions:

    ```bash
    docker-compose up --build
    ```

2.  Access the application at `http://localhost:8000`.

## Project Structure

- `main.py`: Entry point of the application.
- `pyproject.toml`: Python dependencies and project metadata.
- `Dockerfile`: Container definition.
- `docker-compose.yml`: Service definition.
