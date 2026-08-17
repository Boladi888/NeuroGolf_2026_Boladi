# Librarian Claude Lane

Drop durable lesson packets here as:

```text
<UTC>-taskNNN.lesson.md
```

Readers ignore `*.tmp`. Librarian drains this lane with:

```text
python Agenters/Librarian/librarian_ctl.py digest --lane claude
python Agenters/Librarian/librarian_ctl.py show --file <UTC>-taskNNN.lesson.md
python Agenters/Librarian/librarian_ctl.py archive --lane claude --file <UTC>-taskNNN.lesson.md --apply
```
