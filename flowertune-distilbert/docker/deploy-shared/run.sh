echo "Setting up the environment..."
echo "Installing uv..."
pip install --user uv --break-system-packages

if [ -d ".venv" ]; then
    rm -rf .venv
fi

echo "Creating virtual environment..."
uv venv .venv

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Installing dependencies (excluding flwr[simulation])..."
sed 's/.*flwr\[simulation\].*//' pyproject.toml | uv pip install --break-system-packages -r -
# uv pip install -e .

echo "Environment setup complete."

echo "Starting the application using Docker Compose..."

export PROJECT_DIR=$(pwd)
docker compose -f ./docker/deploy-shared/docker-compose.yaml up --build