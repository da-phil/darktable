#!/usr/bin/env bash
#
# launch open source LLMs locally in docker, served by ollama
#
#   ./llm.sh --model qwen3:8b              chat with a model in the terminal
#   ./llm.sh --model mistral serve         keep the API running in the background
#
# see ./llm.sh --help and README.md

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${LLM_MODEL:-llama3.2}"
GPU="${LLM_GPU:-auto}"
CTX="${OLLAMA_CONTEXT_LENGTH:-}"
WEBUI=0
PROMPT=""
CMD=""
MODEL_SET=0
RUN_ARGS=()

_usage()
{
  cat <<EOF
usage: $(basename "$0") [options] [command] [model] [-- ollama run flags]

commands:
  chat      chat with the model in the terminal (default); downloads it first
  serve     download and load the model, then leave the API running
  pull      download the model only
  list      list downloaded models
  ps        list models loaded in memory
  rm        delete a downloaded model (needs an explicit model)
  logs      follow the server log
  update    fetch newer images and restart; new models often need them
  stop      stop and remove the containers; downloaded models are kept

options:
  -m, --model NAME   model to run (default: \$LLM_MODEL or llama3.2), e.g.
                       qwen3:8b, gemma3:12b, mistral, deepseek-r1:8b,
                       hf.co/<user>/<repo>:Q4_K_M for a GGUF on Hugging Face
                     a bare model argument after the command works too
  -g, --gpu MODE     auto (default), nvidia, amd or cpu
  -c, --ctx TOKENS   context window size, e.g. 16384
  -w, --webui        also start Open WebUI on http://localhost:\${WEBUI_PORT:-3000}
  -p, --prompt TEXT  answer a single prompt and exit; stdin works too
  -h, --help         show this help

--gpu and --ctx configure the server: changing them restarts it

examples:
  $(basename "$0") -m qwen3:8b
  $(basename "$0") -m gemma3:12b -p "explain raw demosaicing in two sentences"
  $(basename "$0") -m mistral --ctx 32768 --webui serve
  $(basename "$0") pull deepseek-r1:8b
  $(basename "$0") -m deepseek-r1:8b -- --think=false --verbose
EOF
}

_die()
{
  echo "$(basename "$0"): $*" >&2
  exit 1
}

_need_value()
{
  [ "$2" -ge 2 ] || _die "$1 needs a value (see --help)"
}

_is_command()
{
  case "$1" in
    chat|serve|pull|list|ps|rm|logs|update|stop) return 0 ;;
    *) return 1 ;;
  esac
}

while [ $# -gt 0 ]; do
  case "$1" in
    -m|--model)  _need_value "$1" $#; MODEL="$2"; MODEL_SET=1; shift 2 ;;
    --model=*)   MODEL="${1#*=}"; MODEL_SET=1; shift ;;
    -g|--gpu)    _need_value "$1" $#; GPU="$2"; shift 2 ;;
    --gpu=*)     GPU="${1#*=}"; shift ;;
    -c|--ctx)    _need_value "$1" $#; CTX="$2"; shift 2 ;;
    --ctx=*)     CTX="${1#*=}"; shift ;;
    -p|--prompt) _need_value "$1" $#; PROMPT="$2"; shift 2 ;;
    --prompt=*)  PROMPT="${1#*=}"; shift ;;
    -w|--webui)  WEBUI=1; shift ;;
    -h|--help)   _usage; exit 0 ;;
    --)          shift; RUN_ARGS=("$@"); break ;;
    -*)          _die "unknown option: $1 (see --help)" ;;
    *)
      if [ -z "$CMD" ] && _is_command "$1"; then
        CMD="$1"
      elif [ "$MODEL_SET" = 0 ]; then
        MODEL="$1"
        MODEL_SET=1
      else
        _die "unexpected argument: $1 (see --help)"
      fi
      shift
      ;;
  esac
done

CMD="${CMD:-chat}"
[ -n "$MODEL" ] || _die "the model name is empty"
if [ -n "$CTX" ]; then
  case "$CTX" in
    *[!0-9]*) _die "--ctx takes a number of tokens, got '$CTX'" ;;
  esac
fi

command -v docker >/dev/null 2>&1 || _die "docker is not installed"
docker compose version >/dev/null 2>&1 \
  || _die "the docker compose plugin is missing (docker compose v2 is needed)"

_detect_gpu()
{
  # the nvidia runtime is only there once the NVIDIA Container Toolkit is set up;
  # without it a GPU reservation makes the container fail to start
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia; then
      echo nvidia
      return
    fi
    echo "warning: NVIDIA GPU found but docker has no nvidia runtime;" \
         "install the NVIDIA Container Toolkit to use it. running on CPU" >&2
  elif [ -e /dev/kfd ] && [ -e /dev/dri ]; then
    echo amd
    return
  fi
  echo cpu
}

case "$GPU" in
  auto) GPU="$(_detect_gpu)" ;;
  nvidia|amd|cpu) ;;
  *) _die "--gpu takes auto, nvidia, amd or cpu, got '$GPU'" ;;
esac

COMPOSE=(docker compose --project-directory "$HERE" -f "$HERE/docker-compose.yml")
case "$GPU" in
  nvidia) COMPOSE+=(-f "$HERE/docker-compose.nvidia.yml") ;;
  amd)    COMPOSE+=(-f "$HERE/docker-compose.amd.yml") ;;
esac
# docker compose skips its usual override file once -f is given, so add it back
if [ -f "$HERE/docker-compose.override.yml" ]; then
  COMPOSE+=(-f "$HERE/docker-compose.override.yml")
fi
if [ "$WEBUI" = 1 ]; then
  COMPOSE+=(--profile webui)
fi

if [ -n "$CTX" ]; then
  export OLLAMA_CONTEXT_LENGTH="$CTX"
fi

HOST="${LLM_BIND:-127.0.0.1}"
if [ "$HOST" = "0.0.0.0" ]; then
  HOST=localhost
fi
API="http://$HOST:${LLM_PORT:-11434}"

_running()
{
  "${COMPOSE[@]}" ps --status running --services 2>/dev/null | grep -qx ollama
}

_up()
{
  echo "starting ollama ($GPU)..." >&2
  "${COMPOSE[@]}" up -d --wait ollama
  if [ "$WEBUI" = 1 ]; then
    "${COMPOSE[@]}" up -d webui
    echo "Open WebUI: http://$HOST:${WEBUI_PORT:-3000} (the first start takes a minute)" >&2
  fi
}

# the management commands reuse a running server as it is, so that a plain
# './llm.sh list' does not restart one started with --ctx or --gpu
_ensure_up()
{
  if [ "$WEBUI" = 1 ] || ! _running; then
    _up
  fi
}

_exec()
{
  local tty=()
  if [ ! -t 0 ] || [ ! -t 1 ]; then
    tty=(-T)
  fi
  # ${a[@]+"${a[@]}"} keeps empty arrays working under set -u in bash 3.2 (macOS)
  "${COMPOSE[@]}" exec ${tty[@]+"${tty[@]}"} ollama "$@"
}

case "$CMD" in
  chat)
    _up
    if [ -n "$PROMPT" ]; then
      _exec ollama run ${RUN_ARGS[@]+"${RUN_ARGS[@]}"} "$MODEL" "$PROMPT"
    else
      _exec ollama run ${RUN_ARGS[@]+"${RUN_ARGS[@]}"} "$MODEL"
    fi
    ;;
  serve)
    _up
    _exec ollama pull "$MODEL"
    echo "loading $MODEL..." >&2
    # an empty prompt only loads the model, so the first request is not slow
    _exec ollama run ${RUN_ARGS[@]+"${RUN_ARGS[@]}"} "$MODEL" "" </dev/null >/dev/null
    cat <<EOF

$MODEL is ready
  ollama API:         $API/api
  OpenAI-compatible:  $API/v1   (model name: "$MODEL")

  curl $API/v1/chat/completions -H 'Content-Type: application/json' \\
    -d '{"model": "$MODEL", "messages": [{"role": "user", "content": "hello"}]}'

stop it with: $0 stop
EOF
    ;;
  pull)
    _ensure_up
    _exec ollama pull "$MODEL"
    ;;
  list)
    _ensure_up
    _exec ollama list
    ;;
  ps)
    _ensure_up
    _exec ollama ps
    ;;
  rm)
    [ "$MODEL_SET" = 1 ] || _die "rm needs the model to delete, e.g. '$0 rm mistral'"
    _ensure_up
    _exec ollama rm "$MODEL"
    ;;
  logs)
    "${COMPOSE[@]}" logs -f ollama
    ;;
  update)
    "${COMPOSE[@]}" pull
    _up
    ;;
  stop)
    "${COMPOSE[@]}" --profile webui down
    ;;
esac
