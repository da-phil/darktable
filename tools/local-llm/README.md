# local LLMs in docker

Run open source language models on your own machine. The models are served by
[Ollama](https://ollama.com) in a docker container; `llm.sh` starts it, picks
the GPU setup and takes the model to run as a command line argument.

Nothing leaves your machine once a model is downloaded, and the server only
listens on `127.0.0.1`.

## requirements

- Docker with the compose plugin (`docker compose version` works)
- optional, NVIDIA GPU: the driver and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- optional, AMD GPU: the amdgpu driver with ROCm support (Linux only)
- enough RAM or VRAM for the model: roughly its download size plus 1-2 GB

Docker on macOS cannot use the Apple GPU, so models run on the CPU there.
The native Ollama app is much faster on a Mac.

## quick start

```sh
cd tools/local-llm
./llm.sh --model qwen3:8b            # chat in the terminal, /bye to quit
```

The first run downloads the Ollama image and the model, which takes a while.
Later runs start in seconds.

```sh
./llm.sh -m gemma4                   # another model
./llm.sh gemma4                      # same thing, model as a plain argument
./llm.sh -m mistral -p "write a haiku about raw files"   # one answer, then exit
cat notes.txt | ./llm.sh -m qwen3:8b -p "summarize this"
./llm.sh -m qwen3:8b --webui serve   # browser chat on http://localhost:3000
./llm.sh stop
```

## choosing a model

Any model from the [Ollama library](https://ollama.com/library) works, written
as `name` or `name:size`. Some starting points:

| model              | download | good for                                  |
|--------------------|---------:|-------------------------------------------|
| `qwen3:0.6b`       |   0.5 GB | testing the setup, very weak machines     |
| `llama3.2`         |   2.0 GB | fast general chat, fine on a CPU          |
| `mistral`          |   4.4 GB | general chat                              |
| `qwen2.5-coder:7b` |   4.7 GB | programming                               |
| `qwen3:8b`         |   5.2 GB | general chat and reasoning                |
| `deepseek-r1:8b`   |   5.2 GB | step-by-step reasoning                    |
| `gemma4`           |   6.6 GB | general chat, images as input             |
| `qwen3.5`          |   6.6 GB | reasoning, images as input                |
| `phi4`             |   9.1 GB | reasoning, 16 GB machines                 |
| `gpt-oss:20b`      |  13.8 GB | strong general model, 24 GB GPU or 32 GB RAM |

GGUF files on Hugging Face run directly too:

```sh
./llm.sh -m hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M
```

Downloaded models are kept in the docker volume `local-llm_ollama-models` and
shared by all runs. `./llm.sh list` shows them, `./llm.sh rm <model>` deletes one.

## commands and options

```
./llm.sh [options] [command] [model] [-- ollama run flags]
```

| command  | does                                                        |
|----------|-------------------------------------------------------------|
| `chat`   | chat in the terminal (default); downloads the model first   |
| `serve`  | download and load the model, leave the API running          |
| `pull`   | download the model only                                     |
| `list`   | list downloaded models                                      |
| `ps`     | list models loaded in memory, and whether on GPU or CPU     |
| `rm`     | delete a downloaded model                                   |
| `logs`   | follow the server log                                       |
| `update` | fetch newer images and restart                              |
| `stop`   | stop the containers; downloaded models are kept             |

| option                 | meaning                                                |
|------------------------|--------------------------------------------------------|
| `-m`, `--model NAME`   | model to use (default `$LLM_MODEL`, else `llama3.2`)   |
| `-g`, `--gpu MODE`     | `auto` (default), `nvidia`, `amd` or `cpu`             |
| `-c`, `--ctx TOKENS`   | context window, e.g. `32768`; costs memory             |
| `-w`, `--webui`        | also start [Open WebUI](https://openwebui.com)         |
| `-p`, `--prompt TEXT`  | answer one prompt and exit                             |

Flags after `--` go to `ollama run`, e.g. `-- --think=false --verbose`.

`--gpu` and `--ctx` configure the server, so changing them restarts it.
`list`, `ps`, `rm` and `pull` reuse a running server as it is.

New models sometimes need a newer Ollama than the one you downloaded; run
`./llm.sh update` when a model fails to load.

## using the API from other programs

`./llm.sh -m qwen3:8b serve` leaves the model loaded and prints the endpoints:

- Ollama API: `http://127.0.0.1:11434/api`
- OpenAI-compatible API: `http://127.0.0.1:11434/v1`, model name `qwen3:8b`,
  any API key

Most tools that speak the OpenAI API (editors, agents, Python's `openai`
package) work when pointed at the second URL.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="unused")
reply = client.chat.completions.create(
    model="qwen3:8b",
    messages=[{"role": "user", "content": "hello"}],
)
print(reply.choices[0].message.content)
```

## configuration

Set these in the environment, e.g. in your shell profile:

| variable              | default          | meaning                                       |
|-----------------------|------------------|-----------------------------------------------|
| `LLM_MODEL`           | `llama3.2`       | model when none is given                      |
| `LLM_GPU`             | `auto`           | same as `--gpu`                               |
| `LLM_PORT`            | `11434`          | API port on the host                          |
| `WEBUI_PORT`          | `3000`           | Open WebUI port on the host                   |
| `LLM_BIND`            | `127.0.0.1`      | `0.0.0.0` exposes the API to your network; it has no password |
| `LLM_MODELS_DIR`      | docker volume    | a host path for the models, e.g. `/data/llm`  |
| `OLLAMA_CONTEXT_LENGTH` | Ollama's default | same as `--ctx`                             |
| `OLLAMA_KEEP_ALIVE`   | `5m`             | how long an idle model stays in memory        |
| `OLLAMA_IMAGE`        | `ollama/ollama:latest` | pin a version, e.g. `ollama/ollama:0.40.15` |

For anything else, such as extra mounts, put a `docker-compose.override.yml`
next to `docker-compose.yml`; `llm.sh` picks it up.

## removing everything

```sh
./llm.sh stop
docker volume rm local-llm_ollama-models local-llm_open-webui
docker image rm ollama/ollama:latest ghcr.io/open-webui/open-webui:main
```
