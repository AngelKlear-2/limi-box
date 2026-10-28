# Event sounds

Put your own WAV files here. Limi loads them by these exact names and silently
skips any that are missing, so you can add them one at a time.

| File | Played when |
|---|---|
| `wake.wav` | wake word detected |
| `ack.wav` | request accepted |
| `sleep.wav` | going to sleep |
| `error.wav` | something failed |
| `confirm.wav` | forget confirmation is asked |
| `done.wav` | forget confirmed |
| `mood_happy.wav` | optional, mood changes to happy |
| `mood_sad.wav` | optional, mood changes to sad |
| `mood_angry.wav` | optional, mood changes to angry |

Format: WAV, PCM 16-bit, mono, 16 kHz or 44.1 kHz. Keep them short (under about
1.5 seconds) so they do not delay the conversation. No sounds ship with the
repo, so Limi is silent until you add some.
