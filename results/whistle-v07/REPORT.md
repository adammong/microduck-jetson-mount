# Whistle speech frontend for mounted MicroDuck

Whistle is available as an optional offline speech recognizer. It replaces Whisper's transcription stage; the existing LFM2.5 duck adapter and bounded motion controller still interpret/execute requests. This is a simulation trial, not a hardware deployment or a demonstrated general accuracy improvement.

## Run on the Mac

After the existing simulation and LFM setup:

```sh
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/setup_whistle.py
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/teleop_skills.py --speech-backend whistle --port 8800 --seed 950 --idle-exit 60
```

Open http://127.0.0.1:8800 and press **Speak a command**. Try “walk forward” or “please back up for two seconds, then sit down.” The page shows **Whistle · offline speech**. To return to Whisper, start with `--speech-backend whisper`; Whisper's separate setup is still required. Buttons and typed commands work independently of speech.

The 16,919,407-byte model and native library are downloaded at pinned revisions and checked against known SHA-256 digests. `speech-model.json` records model/engine revisions and installed package versions. Weights/runtime/cache remain under ignored `tmp/`; inference uses explicit local paths, Hugging Face offline mode, and disabled Needle telemetry. No microphone access occurs until the user presses the button and grants browser permission.

## Same-recording comparison

`comparison.json` compares the current faster-whisper tiny.en INT8 configuration against Whistle, both through the actual fresh-process helper used by the server. Each sees the same ten existing synthetic command/request clips, ten copies with seeded additive white noise at 15 dB, and silence/noise/tone controls. No keyword biasing is enabled. Timing includes process startup, model load and transcription, excluding recording, HTTP, LFM and motion execution. Models already reside on disk; this is not a download/cold-disk benchmark.

| Measurement | Whisper tiny.en INT8 | Whistle |
|---|---:|---:|
| Clean transcript exact and accepted | 9/10 | 6/10 |
| Clean transcript exact, regardless of score rejection | 9/10 | 7/10 |
| Clean median process wall time | 400 ms | 53 ms |
| Noisy transcript exact and accepted | 6/10 | 6/10 |
| Noisy transcript exact, regardless of score rejection | 7/10 | 7/10 |
| Noisy median process wall time | 393 ms | 53 ms |
| Non-speech controls refused | 3/3 | 3/3 |
| Non-speech controls returning empty text | 1/3 | 3/3 |

“Exact” ignores case and punctuation, but is a transcript metric, not proof of correct action understanding. The set includes a negated request and an unrelated phrase. Whistle heard “sit down” as “Sidon” and “stop” as “Stob”; the latter and its otherwise correct “turn left” were refused by score checks. Both models had errors with added white noise. Whisper hallucinated text for noise/tone, but its existing score checks refused both. There is no conclusion about real microphones, accents, motors, room acoustics or population-level accuracy.

Whistle has word probabilities instead of Whisper's segment log probabilities/no-speech scores. The prototype requires nonempty text, every word probability at least 0.2, and a geometric mean at least 0.6. These thresholds are uncalibrated; they cannot establish safety or be compared numerically with Whisper's scores. An exploratory keyword-bias call corrected short spellings but left low word scores, so biasing was not enabled and thresholds were not relaxed. Stop buttons/Space remain available. Speech can omit a negation; guards do not recover information the recognizer lost.

## Controller checks

`http-verification.json` records three actual voice requests and complete policy execution:

- “Walk forward”: forward for three seconds.
- “Please back up for two seconds, then sit down”: backward → seated.
- “Could you stand up and turn right for two seconds”: stand → right turn.

All three completed without a controller fault. Clean/noisy negated speech, unrelated speech, noise, tone, silence and an invalid WAV were rejected. Stop and Reset superseded in-flight speech/LFM requests; neither stale plan executed. The selected walking/posture policy SHA-256 hashes remain unchanged. Shared invalid-format, duration, truncated-WAV and silence checks passed, as did browser PCM resampling checks. These are narrow integration checks, not new physics training or a repeat of the entire prior robustness study.

Reproduce the speech comparison and input checks:

```sh
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/evaluate_speech.py
tmp/microduck-lab/microduck_local/.venv/bin/python training/verify_speech.py
node training/verify_voice.cjs results/whistle-v07/browser-audio-verification.json
```

For the HTTP probe, run a separate server with `--speech-backend whistle --port 8801 --seed 950 --no-render`, then run `tmp/whistle-venv/bin/python results/whistle-v07/http_probe.py` from the repository root. The probe controls that simulator, so use a separate instance.

## Limits

Human microphone, physical duck and Jetson measurements are untested. The setup deliberately targets the tested native Apple Silicon runtime; it does not install a Jetson runtime. Whistle's model file size is not a measurement of runtime RAM. The existing LFM layer's 14/24 supported-request result is unchanged, and language interpretation still dominates multistep response time. No streaming/wake-word interface was added; the browser retains its explicit six-second maximum utterance and existing watchdog/plan guards.
