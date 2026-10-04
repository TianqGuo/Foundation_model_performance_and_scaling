# Stage 3 matched comparison

Single-seed smoke/pilot; no superiority claim. Different clipping bounds explicitly allowed; exploratory comparison. Sequence normalization and eager inference; this is not a tuned GSPO replication.

| Method | Clip low/high | Final accuracy | Mean sync (s) | Mean rollout (s) |
|---|---|---:|---:|---:|
| grpo | 0.2/0.2 | 0.0938 | 0.279 | 3.792 |
| gspo | 0.0003/0.0004 | 0.0938 | 0.245 | 3.781 |
