#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_EZPC_DIR="${SCRIPT_DIR}"

IMAGE_NAME="${IMAGE_NAME:-ezpc-gpu-mpc:cuda11.8}"
CONTAINER_NAME="${CONTAINER_NAME:-ezpc-gpu-mpc}"
DOCKERFILE_PATH="${DOCKERFILE_PATH:-${HOST_EZPC_DIR}/GPU-MPC/Dockerfile_Gen}"
CONTAINER_EZPC_DIR="${CONTAINER_EZPC_DIR:-/workspace/EzPC}"
CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-/workspace/EzPC/GPU-MPC}"

usage() {
  cat <<EOF
Usage: $0 <command>

Commands:
  build    Build Docker image for GPU-MPC
  create   Create persistent container (if it does not exist)
  start    Start existing container
  enter    Enter container shell (starts container if needed)
  stop     Stop container
  restart  Restart container
  status   Show container status
  logs     Show container logs

Examples:
  $0 build
  $0 create
  $0 enter
EOF
}

require_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "Error: docker not found"
    exit 1
  fi
}

container_exists() {
  docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1
}

container_running() {
  [ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null || true)" = "true" ]
}

build_image() {
  require_docker
  if [ ! -f "${DOCKERFILE_PATH}" ]; then
    echo "Error: Dockerfile not found: ${DOCKERFILE_PATH}"
    exit 1
  fi

  docker build \
    -f "${DOCKERFILE_PATH}" \
    -t "${IMAGE_NAME}" \
    "${HOST_EZPC_DIR}"
}

create_container() {
  require_docker
  if container_exists; then
    echo "Container ${CONTAINER_NAME} already exists."
    return 0
  fi

  docker run -dit \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    --net=host \
    -v "${HOST_EZPC_DIR}:${CONTAINER_EZPC_DIR}" \
    -w "${CONTAINER_WORKDIR}" \
    "${IMAGE_NAME}" \
    bash -lc "while true; do sleep 3600; done"

  echo "Created container ${CONTAINER_NAME}."
}

start_container() {
  require_docker
  if ! container_exists; then
    echo "Container ${CONTAINER_NAME} does not exist. Run: $0 create"
    exit 1
  fi

  if container_running; then
    echo "Container ${CONTAINER_NAME} is already running."
  else
    docker start "${CONTAINER_NAME}" >/dev/null
    echo "Started container ${CONTAINER_NAME}."
  fi
}

enter_container() {
  start_container
  docker exec -it "${CONTAINER_NAME}" bash
}

stop_container() {
  require_docker
  if container_running; then
    docker stop "${CONTAINER_NAME}" >/dev/null
    echo "Stopped container ${CONTAINER_NAME}."
  else
    echo "Container ${CONTAINER_NAME} is not running."
  fi
}

show_status() {
  require_docker
  docker ps -a --filter "name=^${CONTAINER_NAME}$"
}

show_logs() {
  require_docker
  docker logs "${CONTAINER_NAME}"
}

cmd="${1:-}"
case "${cmd}" in
  build) build_image ;;
  create) create_container ;;
  start) start_container ;;
  enter) enter_container ;;
  stop) stop_container ;;
  restart) stop_container; start_container ;;
  status) show_status ;;
  logs) show_logs ;;
  *) usage; exit 1 ;;
esac
