#!/usr/bin/env bash
set -euo pipefail

# One-time interactive setup. The key is never echoed and is stored outside the project.
config_dir="${XDG_CONFIG_HOME:-$HOME/.config}/claude-worker"
env_file="$config_dir/env"
bashrc_file="$HOME/.bashrc"
source_line='[ -r "$HOME/.config/claude-worker/env" ] && . "$HOME/.config/claude-worker/env"'

read -r -s -p "Claude API key: " api_key
printf '\n'
if [[ -z "$api_key" ]]; then
  echo "No key entered; nothing changed." >&2
  exit 1
fi

umask 077
mkdir -p "$config_dir"
printf 'export ANTHROPIC_API_KEY=%q\n' "$api_key" > "$env_file"
unset api_key
chmod 600 "$env_file"

if ! grep -Fqx "$source_line" "$bashrc_file" 2>/dev/null; then
  temporary="${bashrc_file}.claude-worker-tmp"
  {
    printf '%s\n' "$source_line"
    [[ -f "$bashrc_file" ]] && cat "$bashrc_file"
  } > "$temporary"
  chmod --reference="$bashrc_file" "$temporary" 2>/dev/null || chmod 600 "$temporary"
  mv "$temporary" "$bashrc_file"
fi

echo "Claude credential configured outside the project with mode 600."
echo "Reconnect the Codex SSH workspace once so the app process inherits it."
