# ProjectMinima Win32 desktop frontend

The frontend is a separate native Win32 application written in C. It launches `Minima.exe` as a child process and communicates through standard input/output plus small filesystem signals. The UI does not link into or execute model/training code. Training and inference remain in the engine process, so the window stays responsive.

## Build

1. Open `Minima.sln` in Visual Studio 2022.
2. Select the x64 platform.
3. Build the `Minima` and `MinimaGUI` projects.
4. Keep `Minima.exe` and `MinimaGUI.exe` in the same output directory.
5. Launch `MinimaGUI.exe`.

The GUI saves its settings and corpus-file list in `minima_gui.ini` beside the executables, and restores them on the next launch.

The engine working directory is the executable directory. Keep the `data` folder and checkpoint/tokenizer files in the locations expected by the engine.

## Training controls

- Add/remove multiple corpus files.
- Set additional optimizer steps and choose whether to resume.
- Restore the standard model, training, tokenizer, and generation settings with **Basic defaults**; selected corpus files are preserved.
- Configure vocabulary size, embedding dimension, layers, attention heads, FFN dimension, and model context length.
- Configure batch size and training sequence length.
- Configure learning rate, warmup, and cosine or constant learning-rate schedule.
- Configure training-log interval, validation interval, validation batch count, and checkpoint interval independently.
- Choose the random seed.
- Configure tokenizer sample byte limit (4 MiB maximum) and either evenly spread sampling or prefix sampling.
- Start training, request a graceful stop, or request an immediate checkpoint save.

Runtime model limits are validated by the engine: vocabulary 260-8192, dimension 1-512, layers 1-16, heads 1-32 (dimension must divide evenly by heads), FFN dimension 1-1024, and context length 32-512. Batch size is 1-16 and training sequence length is 8-512 and cannot exceed context length. The engine also rejects excessively large logits scratch configurations.

Changing vocabulary or model architecture makes an existing checkpoint incompatible. Resume and chat must use the same model configuration and tokenizer as the checkpoint.

## Results and monitoring

The GUI displays engine output, including:

- Training and validation loss.
- Learning rate and optimizer step.
- Training and validation perplexity.
- Total optimizer steps and elapsed training CPU time.
- Parameter count and weight, optimizer, and gradient memory estimates.
- Estimated logits scratch and KV-cache memory.
- Tokenizer vocabulary, merge count, and token-length statistics.
- Numerical gradient-check results.
- Checkpoint save and compatibility status.

The engine appends structured rows to `minima_metrics.csv`. The lightweight GDI graph reads that CSV and plots training and validation loss. It refreshes once per second and does not participate in training.

## Chat

- Start chat loads the existing checkpoint and tokenizer.
- The message field accepts typed text; its dropdown contains example prompts. Press Enter to start chat and send the first message, or click Start chat and send when the engine is ready. Enter sends later messages.
- Resume is cleared automatically after a successful tokenizer rebuild because the old checkpoint is incompatible with the new tokenizer.
- Temperature, top-k, top-p, repetition penalty, and maximum new tokens are configurable.
- Chat transcript is appended to `minima_chat.txt` and reloaded when the GUI starts. The visible transcript persists; the model's rolling context starts fresh when a new chat process is launched.
- Stop generation terminates the inference child process. This does not affect checkpoint weights.

Chat requires `minima.model` and `minima.bpe`. Set the model configuration controls to match the checkpoint before starting chat if it was trained with non-default architecture settings.

## Process and persistence behavior

- Training output is redirected to the GUI log and flushed by the engine.
- Training runs in a separate process.
- Stop training creates `minima.stop`; the engine finishes its current optimizer step, evaluates, saves a checkpoint, and exits.
- Save checkpoint creates `minima.save`; the engine saves at the next optimizer-step boundary.
- Automatic checkpoints are written at the configured interval and at the end of a training segment.
- `minima_metrics.csv`, `minima_chat.txt`, and signal files are stored beside the executables.

A Windows x64 GitHub Actions workflow builds both executables and publishes them as a downloadable artifact for each successful build.

The UI is deliberately a controller. It must not own model weights, perform forward/backward passes, or block the window thread with training or generation work.


## GUI module boundaries

The Win32 shell is being split into focused modules while keeping the existing window procedure and control behavior intact:

- `gui_layout.c/.h` owns responsive control positioning, tab visibility, and DPI scaling.
- `gui_pages.c/.h` owns the tab strip, top-level page containers, and the Training, Advanced, and Chat control construction.
- `gui_actions.c/.h` owns command-ID dispatch and routes commands to the application action callbacks.
- `MinimaGUI.c` retains process orchestration, application state, and the concrete callbacks that connect UI actions to settings and engine operations.

This is an incremental extraction: concrete action callbacks and application/process state remain in the application file, while the page controls, layout, and command routing now sit behind dedicated interfaces. The goal is to keep each extraction behavior-preserving and independently buildable.
