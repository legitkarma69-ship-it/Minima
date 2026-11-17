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
- Choose AdamW or Lion for fresh runs, configure weight decay, and set OpenMP worker threads (0 selects the runtime default). Resume automatically retains the optimizer type stored in the checkpoint; weight decay and thread count remain configurable.
- Set a Windows Job Object hard CPU cap (0 disables it; 1-100 is a percentage of total logical-processor capacity), a hard process-group committed-memory cap in GiB (0 disables it; 1-64 GiB), and a separate shared engine allocation budget in GiB (0 disables it; 1-64 GiB). Limits are applied before the engine starts, including chat and tokenizer rebuilds. The 24-question evaluation process tree receives the same limits.
- Configure training-log interval, validation interval, validation batch count, checkpoint interval, and previous-checkpoint retention independently. When a fresh GUI training run starts, the previous `minima_metrics.csv` and its metadata sidecar are automatically archived with a timestamp instead of being deleted. This preserves the prior run for comparison. Each training invocation also writes a `minima_metrics.csv.meta.json` sidecar (next to the selected metrics file) containing model dimensions, optimizer hyperparameters, schedule, batch/sequence settings, seed, memory budget, corpus mix, and per-source weights. Use **Compare runs** on the Logs page to select two or more metrics CSVs; Minima ranks them by best validation loss, writes `minima_run_comparison.csv` in the repository root, and opens it when finished. This requires Python 3 and the Windows `py` launcher.
- Choose the random seed.
- Configure tokenizer sample byte limit (4 MiB maximum) and either evenly spread sampling or prefix sampling.
- Start training, request a graceful stop, or request an immediate checkpoint save.

Runtime model limits are validated by the engine: vocabulary 260-32768, dimension 1-512, layers 1-16, heads 1-32 (dimension must divide evenly by heads), FFN dimension 1-1024, and context length 32-512. Batch size is 1-16 and training sequence length is 8-512 and cannot exceed context length. The default 4,096-vocabulary preset uses dimension 128, 2 layers, 4 attention heads, FFN 256, context 128, batch 1, and sequence length 128 (about 819,840 parameters). The default training schedule is 100,000 optimizer steps, beginning with a 2,000-step segment; continue with an 8,000-step segment to reach 10,000 cumulative steps. On first launch after this defaults migration, the GUI preserves the selected corpus list and unrelated settings, disables Resume, and configures the tokenizer controls for a fresh rebuild using the 4 MiB spread sample. Rebuilding the tokenizer invalidates compatibility with any existing checkpoint; train a fresh model afterward. The engine also rejects excessively large logits scratch configurations.

Changing vocabulary or model architecture makes an existing checkpoint incompatible. Resume and chat must use the same model configuration and tokenizer as the checkpoint.

## Results and monitoring

Resource profiles are available as **Eco**, **Balanced**, **Performance**, **Training**, and **Chat**. Advanced settings also accept optional comma-separated source multipliers in Corpus list order, such as `1,0.5,2`; each non-empty corpus file needs one positive multiplier. Fresh training runs also expose AdamW/Lion beta1, beta2, and epsilon values; selecting Lion fills its usual beta defaults, while Resume restores the optimizer hyperparameters stored in the checkpoint. The **Generation** tab groups temperature, top-k/top-p/min-p, repetition/frequency/presence penalties, no-repeat n-gram, min/max new tokens, sampling seed, stop sequence, and the one-shot Generation Lab. Stop text is removed from the generated response; min-new-tokens delays stop matching until the requested minimum is reached. Normal chat reads these settings when the chat process starts; the Generation Lab button applies the current values as a one-shot override to an active chat. These multiply the selected global Data mix weights. Use **Save profile** to export the current GUI configuration to an INI file, or **Load profile** to import one and apply its settings immediately. They set CPU cap, Windows committed-memory cap, arena budget, OpenMP threads, and process priority. Editing one of those values changes the profile to Custom. The presets are starting points, not hardware detection or guarantees.

The Monitor panel displays live process-group CPU utilization, working-set RAM, and private committed memory, along with per-run CPU and memory peaks. During training, the Pause training button pauses between optimizer steps without terminating Minima; Resume training removes the pause signal. Progress logs include cumulative tokens, steps/second, tokens/second, current learning rate, and estimated time remaining. They also report the best validation loss and step observed in the current training segment, saving that checkpoint as `minima.segment_best.model`. Set **Keep checkpoints** in Advanced to retain 0-20 previous versions of `minima.model` as `.prev.N` archives before fresh, scheduled, manual, and final-segment saves; the current checkpoint remains in place, and the best-validation checkpoint is kept separately. The default is 3, and 0 disables archive creation. A resumed segment starts a new best-loss comparison rather than carrying the previous segment's best record forward. Configure CPU and RAM warning thresholds in the same panel; each threshold logs one warning per run when reached. CPU and RAM caps are applied by Windows Job Objects. The Windows memory cap is a hard job-wide committed-memory limit, not a physical working-set cap. The engine budget is a separate soft allocation ceiling shared by permanent, temporary, and corpus arena pages plus tracked tokenizer and checkpoint heap allocations. Allocation failures are reported before the requested allocation proceeds. Disk-backed token-ID mappings and Windows/runtime overhead are estimated separately; the Windows Job Object cap remains the process-wide backstop. The engine prints a pre-training estimate combining its observed budgeted high-water mark, mapped token-file capacity, and a 256 MiB runtime planning reserve. A value of 0 disables either cap. CPU thread count and CPU utilization are separate controls: limiting threads does not impose a CPU percentage ceiling.

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

## Chat and evaluation

- The **Chat** tab contains only the conversation transcript and message controls. The **Logs** tab contains the complete engine/evaluation output, including training steps, validation, warnings, and errors.
- Start chat loads the existing checkpoint and tokenizer.
- The message field accepts typed text; its dropdown contains example prompts. Press Enter to start chat and send the first message, or click Start chat and send when the engine is ready. Enter sends later messages.
- Resume is cleared automatically after a successful tokenizer rebuild because the old checkpoint is incompatible with the new tokenizer.
- Temperature, top-k, top-p, repetition penalty, and maximum new tokens are configurable.
- Chat transcript is appended to `minima_chat.txt` and reloaded when the GUI starts. **Clear chat** clears the visible and saved transcript and sends `/clear` to an active chat session so its context is reset too.
- **Clear logs** clears the visible Logs tab only. It does not delete metrics, checkpoints, or evaluation results.
- Stop generation terminates the inference child process. This does not affect checkpoint weights.

Chat requires `minima.model` and `minima.bpe`. Set the model configuration controls to match the checkpoint before starting chat if it was trained with non-default architecture settings.

### 24-question evaluation

- Click **Run 24-question test** in the Chat tab to run the held-out `data/evaluation/logic.jsonl` set.
- The GUI launches `tools/evaluate_logic.py` through the Windows `py -3` launcher. Python 3 and the repository's `tools` and `data/evaluation` folders must be available beside the x64 build folder.
- The runner loads the checkpoint once, clears model conversation context between prompts, uses greedy decoding, and prints each question and answer to Logs as soon as it completes.
- Results are written incrementally to `logic_results.jsonl` beside the executables. Reference answers and scoring notes are never sent to Minima.
- **Stop evaluation** terminates the evaluation process and its Minima child process. Completed result records remain on disk.

## Process and persistence behavior

- Training output is redirected to the GUI log and flushed by the engine.
- Training runs in a separate process.
- Stop training creates `minima.stop`; the engine finishes its current optimizer step, evaluates, saves a checkpoint, and exits.
- Save checkpoint creates `minima.save`; the engine saves at the next optimizer-step boundary.
- Automatic checkpoints are written at the configured interval and at the end of a training segment.
- Long-run progress, validation, and checkpoint intervals default to 1,000 steps. Auto settings scale them with the planned run length, and the Advanced page keeps each interval editable.
- `minima_metrics.csv`, `minima_chat.txt`, and signal files are stored beside the executables.
- Resource limits are persisted in `minima_gui.ini`. The CPU cap and Windows committed-memory cap are applied to each newly launched engine/evaluation process group; the engine allocation budget is passed to Minima and applies to its arenas plus tracked tokenizer/checkpoint heap allocations. CPU and RAM warning thresholds are also persisted and trigger at most one log warning per run. Changing the fields does not alter an already-running job.

A Windows x64 GitHub Actions workflow builds both executables and publishes them as a downloadable artifact for each successful build.

The UI is deliberately a controller. It must not own model weights, perform forward/backward passes, or block the window thread with training or generation work.


## GUI module boundaries

The Win32 shell is being split into focused modules while keeping the existing window procedure and control behavior intact:

- `gui_layout.c/.h` owns responsive control positioning, tab visibility, and DPI scaling.
- `gui_pages.c/.h` owns the tab strip, top-level page containers, and the Training, Advanced, Chat, and Logs control construction.
- `gui_actions.c/.h` owns command-ID dispatch and routes commands to the application action callbacks.
- `MinimaGUI.c` retains process orchestration, application state, and the concrete callbacks that connect UI actions to settings and engine operations.

This is an incremental extraction: concrete action callbacks and application/process state remain in the application file, while the page controls, layout, and command routing now sit behind dedicated interfaces. The goal is to keep each extraction behavior-preserving and independently buildable.
