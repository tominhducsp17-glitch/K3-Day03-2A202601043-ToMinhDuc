# Security Scan

Command:

```powershell
rg -n "sk-|AIza" . --glob "*.py" --glob "*.md" --glob "*.json" --glob "!.env" --glob "!.venv/**" --glob "!logs/**" --glob "!**/__pycache__/**"
```

Result:

- No API key was found in source code, tests, reports, or artifacts.
- The only match was the literal example pattern inside `day3-lab-chatbot-vs-react-agent.md`.
- `.env` is ignored by `.gitignore` and should not be committed.

Submission safety status: pass.
