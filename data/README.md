# Training text

Place UTF-8 plain-text material in these files:

- `books.txt`: books, essays, stories, documentation, and other long-form text.
- `chatlogs.txt`: conversation logs in plain text.

Keep the two sources separate so they can be loaded, sampled, and weighted independently later.

Use ordinary text. No JSON, markup wrappers, or special delimiters are required. Separate documents with a blank line. For chat logs, a readable format such as:

User: ...
Assistant: ...

is fine.

Important: these files are corpus placeholders. The current smoke-test program still trains on its built-in synthetic sample; file loading and dataset mixing have not yet been connected.
