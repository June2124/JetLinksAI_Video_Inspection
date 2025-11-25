#!/usr/bin/env bash
# JetLinksAI starter (Uvicorn/FastAPI)
# Location: /opt/jetlinks-ai/start.sh

set -Eeuo pipefail

APP_DIR="/data/JetLinksAI-Video-Inspection/JetLinksAI"
APP_MODULE="app_service:app"           
CONDA_ENV_BIN="/home/linaro/miniconda3/envs/JetLinksAI-Video-Inspection/bin" # 本系统存放的虚拟环境路径

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-1}"                # 必须=1，因为我们有全局 JOBS / 本地VLM 实例

BASE_DIR="/data/JetLinksAI-Video-Inspection"
LOG_DIR="${BASE_DIR}/logs"
RUN_DIR="${BASE_DIR}/run"
ENV_FILE="${BASE_DIR}/.env"
PID_FILE="${RUN_DIR}/uvicorn.pid"
LOG_OUT="${LOG_DIR}/uvicorn.out"
LOG_ERR="${LOG_DIR}/uvicorn.err"

ensure_dirs() {
  mkdir -p "$LOG_DIR" "$RUN_DIR"
}
 
load_env() {
  if [[ -f "$ENV_FILE" ]]; then
    set -a
    source <(grep -E '^[A-Za-z_][A-Za-z0-9_]*=' "$ENV_FILE" || true)
    set -a
  fi
}

is_running() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  pid=$(cat "$PID_FILE" 2>/dev/null || echo "")
  [[ -n "$pid" && -d "/proc/$pid" ]]
}

start() {
  ensure_dirs
  load_env

  if is_running; then
    echo "Already running. pid=$(cat "$PID_FILE")"
    exit 0
  fi

  if [[ ! -x "${CONDA_ENV_BIN}/uvicorn" ]]; then
    echo "ERROR: ${CONDA_ENV_BIN}/uvicorn 不存在或不可执行"
    exit 1
  fi
  if [[ ! -d "$APP_DIR" ]]; then
    echo "ERROR: APP_DIR 不存在: $APP_DIR"
    exit 1
  fi

  echo "Starting JetLinksAI on ${HOST}:${PORT} (workers=${WORKERS}) ..."
  cd "$APP_DIR"
  nohup "${CONDA_ENV_BIN}/uvicorn" "$APP_MODULE" \
    --host "$HOST" --port "$PORT" --workers "$WORKERS" \
    --proxy-headers --forwarded-allow-ips="*" \
    >>"$LOG_OUT" 2>>"$LOG_ERR" &

  echo $! > "$PID_FILE"
  sleep 0.5
  echo "Started. pid=$(cat "$PID_FILE")"
  echo "Logs: $LOG_OUT"
}

stop() {
  if ! is_running; then
    echo "Not running."
    rm -f "$PID_FILE"
    exit 0
  fi
  local pid
  pid=$(cat "$PID_FILE")
  echo "Stopping pid=$pid ..."
  kill "$pid" 2>/dev/null || true

  for _ in {1..20}; do
    is_running || break
    sleep 0.2
  done

  if is_running; then
    echo "Force killing pid=$pid"
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
  echo "Stopped."
}

restart() {
  stop || true
  start
}

status() {
  if is_running; then
    echo "Running. pid=$(cat "$PID_FILE")"
  else
    echo "Stopped."
  fi
}

tail_log() {
  ensure_dirs
  touch "$LOG_OUT" "$LOG_ERR"
  echo "=== tail -f $LOG_OUT $LOG_ERR ==="
  tail -n 200 -f "$LOG_OUT" "$LOG_ERR"
}

setkey() {
  ensure_dirs
  read -r -s -p "DashScope API Key: " DKEY; echo
  if [[ -z "$DKEY" ]]; then
    echo "Empty key, aborted."
    exit 1
  fi
  if [[ -f "$ENV_FILE" ]]; then
    sed -i '/^DASHSCOPE_API_KEY=/d' "$ENV_FILE"
  fi
  printf 'DASHSCOPE_API_KEY=%s\n' "$DKEY" >> "$ENV_FILE"
  chmod 600 "$ENV_FILE"
  # 如果盒子登录用户不是 linaro，可以改成相应用户；留着 || true 避免报错阻塞
  chown linaro:linaro "$ENV_FILE" 2>/dev/null || true
  echo "Saved to $ENV_FILE"
}

usage() {
  cat <<EOF
Usage: $0 {start|stop|restart|status|tail|setkey}
  start    启动服务
  stop     停止服务
  restart  重启服务
  status   运行状态
  tail     实时查看日志
  setkey   交互式写入 DashScope API Key 到 $ENV_FILE
Environment:
  HOST=${HOST}  PORT=${PORT}  WORKERS=${WORKERS}
  APP_DIR=$APP_DIR
  ENV_FILE=$ENV_FILE
Logs:
  $LOG_OUT
  $LOG_ERR
EOF
}

cmd="${1:-usage}"
case "$cmd" in
  start)   start ;;
  stop)    stop ;;
  restart) restart ;;
  status)  status ;;
  tail)    tail_log ;;
  setkey)  setkey ;;
  *)       usage ;;
esac
