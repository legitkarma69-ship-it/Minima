# ProjectMinima Win32 frontend

This is the first UI integration layer. It is a separate native Win32 C application and launches the existing `Minima.exe` engine as a child process. It does not link UI code into the model or training implementation.

## Build

1. Open the repository's `Minima.sln` in Visual Studio 2022.
2. Select `x64` and build both `Minima` and `MinimaGUI`.
3. Keep `Minima.exe` and `MinimaGUI.exe` in the same output directory.
4. Launch `MinimaGUI.exe`.

The GUI starts the engine with its own directory as the working directory. Keep the existing data folder and checkpoint/tokenizer files in the locations expected by the engine.

## Implemented in this foundation

- Native Win32 controls only; no external UI framework.
- Multi-select corpus file picker and remove-selected action.
- Additional training-step input.
- Resume existing checkpoint option.
- Start and stop controls.
- Background child process; the UI thread remains responsive.
- Redirected engine stdout/stderr displayed in a live log.
- Engine stdout/stderr are unbuffered so progress reaches the GUI promptly.
- Exit status reporting.

## Not yet wired to runtime engine APIs

The current engine still defines model architecture, batch/sequence sizes, tokenizer target, and several training settings as compile-time constants. This frontend intentionally does not display fake controls for settings that the engine cannot yet honor.

The following require the next engine refactor before they can be exposed as functional UI controls:

- Runtime vocabulary/model architecture configuration.
- Tokenizer sample size and sampling strategy configuration.
- Learning-rate schedule configuration, validation cadence, checkpoint cadence, and random seed.
- Structured training metric callbacks and plotted loss/perplexity charts.
- Structured model/tokenizer/checkpoint/gradient diagnostics.
- In-process or separate-process interactive inference, chat history, cancellation, top-p and repetition penalty.

The stop button currently force-terminates the child process after confirmation. The engine's most recent periodic checkpoint remains usable, but steps since that checkpoint can be lost. A cooperative cancellation request and final checkpoint are planned before treating Stop as a graceful operation.

## Architecture rule

The UI is a controller, not part of the model. Future communication should use a small engine API or a versioned local IPC protocol. Training/inference must remain on worker threads or in a separate process; the UI thread must never execute model work.
