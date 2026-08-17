env -u ANTHROPIC_API_KEY CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=0 claude -p "$(cat Agenters/Orchestrator/runs/beta_p8c_claude_cli/launch_packet.md)" \
  --model claude-sonnet-4-6 \
  --output-format json \
  --dangerously-skip-permissions \
  < /dev/null > Agenters/Orchestrator/runs/beta_p8c_claude_cli/out.json 2>&1
