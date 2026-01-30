# The Simulation Loop

**The Simulation Loop** is the core primitive of `xrtm-train`. It allows us to optimize agents by replaying them against history.

## Architecture

The simulation loop consists of three phases:

1.  **Fetch (History)**: The `Backtester` requests a set of questions from `xrtm-data` with a specific `snapshot_time`.
2.  **Act (Agent)**: The Agent (from `xrtm-forecast`) produces a prediction using ONLY that historical context.
3.  **Score (Future)**: The `Backtester` resolves the question using the "Future" resolution from `xrtm-data` and scores it with `xrtm-eval`.

## Strict Isolation

The most critical rule of the simulation loop is **Temporal Isolation**.

- The Agent MUST NOT have access to the internet *after* the `snapshot_time`.
- The `TraceReplayer` allows debugging a specific run by re-instantiating the agent's memory state at that exact moment in history.
