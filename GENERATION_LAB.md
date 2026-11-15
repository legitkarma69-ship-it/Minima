# Win32 Generation Lab

The Generation Lab is available while a checkpoint chat session is running.

1. Start chat from the GUI. The model checkpoint and tokenizer are loaded once.
2. Expand **Advanced / generation** to edit temperature, top-k, top-p,
   repetition penalty, and maximum new tokens.
3. Enter a prompt in the bottom prompt field, or choose one of the built-in
   prompts from its dropdown.
4. Click **Test settings**. Minima generates one isolated response and prints
   it in the existing output/chat history panel, alongside the exact settings
   and prompt used.

A test applies its sampling settings to exactly one generation. It does not
change the active chat defaults, model weights, optimizer state, or checkpoint,
and it does not add the test prompt or response to the model's rolling chat
context. The next ordinary **Generate / Send** message uses the chat settings
that were active when the chat process started.

The sampling controls remain editable during chat so successive tests can be
compared without restarting the model. The prompt dropdown includes a few
fixed continuations and conversational prompts; custom prompts can be typed
into the same editable field.

Supported ranges:
- Temperature: 0.05 to 5.0
- Top-k: 1 to 256
- Top-p: 0.01 to 1.0
- Repetition penalty: 1.0 to 3.0
- Maximum new tokens: 1 to 512 (actual output is also limited by remaining
  model context)

The chat protocol uses a private `/test` control line followed by the prompt.
Invalid overrides are rejected, and the chat session returns to its ready state.
