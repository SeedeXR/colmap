# Bash completion for the `colmap` CLI.
#
# Install: source this file from your ~/.bashrc, or copy it to
#   $(brew --prefix)/etc/bash_completion.d/colmap   (Homebrew bash-completion)
#   /usr/share/bash-completion/completions/colmap   (system-wide on Linux)
#
# Commands are queried from `colmap help` at completion time, so the list never
# goes stale as commands are added/removed.

_colmap_commands() {
  colmap help 2>/dev/null | awk '
    /Available commands:/ { in_cmds = 1; next }
    in_cmds && NF { sub(/^[[:space:]]+/, ""); print $1 }'
}

_colmap() {
  local cur prev cword
  cur="${COMP_WORDS[COMP_CWORD]}"
  prev="${COMP_WORDS[COMP_CWORD-1]}"
  cword=$COMP_CWORD

  # First argument: the subcommand.
  if [[ $cword -eq 1 ]]; then
    COMPREPLY=( $(compgen -W "$(_colmap_commands)" -- "$cur") )
    return 0
  fi

  # Options that take a path -> complete files/dirs.
  case "$prev" in
    --database_path|--image_path|--project_path|--output_path|--input_path|--path|--workspace_path|--vocab_tree_path|--export_path|--import_path)
      compgen -f -- "$cur" >/dev/null 2>&1 && COMPREPLY=( $(compgen -f -- "$cur") )
      return 0;;
    --progress_format)
      COMPREPLY=( $(compgen -W "none plain jsonl" -- "$cur") ); return 0;;
    --format)
      COMPREPLY=( $(compgen -W "text json" -- "$cur") ); return 0;;
  esac

  # Common global flags (a representative subset; each command's full list is
  # available via `colmap <command> --help`).
  if [[ "$cur" == -* ]]; then
    COMPREPLY=( $(compgen -W "\
--help --project_path --database_path --image_path --output_path \
--progress_format --progress_every --quiet --log_level --log_target \
--FeatureExtraction.use_gpu --FeatureExtraction.num_threads \
--FeatureExtraction.max_memory_gb --SiftExtraction.first_octave \
--FeatureMatching.use_gpu" -- "$cur") )
    return 0
  fi

  # Otherwise default to file completion.
  COMPREPLY=( $(compgen -f -- "$cur") )
  return 0
}

complete -o filenames -F _colmap colmap
