screen -S llama-server -X quit 2>/dev/null

if ! command -v llama >/dev/null 2>&1; then
  echo "Installing llama.cpp"
  curl -LsSf https://llama.app/install.sh | sh
fi

screen -dmS llama-server llama serve --host 0.0.0.0 --models-preset $(dirname $0)/config.ini
