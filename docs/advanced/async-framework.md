# Async Framework Architecture

This document explains the async scheduling architecture that enables efficient GPU utilization when running multi-agent conversations at scale.

## Motivation

Running multi-agent conversations naively would result in poor GPU utilization:

1. **Sequential within conversation**: If agents respond one after another within a conversation, the GPU sits idle between requests
1. **Sequential across conversations**: If we process one conversation at a time, we lose massive parallelism opportunities
1. **Dependency constraints**: Some roles (e.g., moderator) naturally need to wait for other agents before responding
1. **Different number of retries**: A message may need several retries (especially for smaller models), while processed requests need to wait under static batching

The async framework solves these problems by:

- **Parallelizing across conversations**: Different conversations have no cross-visibility and can run fully in parallel
- **Parallelizing within rounds**: Within a single round, agents without dependencies can generate responses concurrently
- **Maximizing GPU batch utilization**: vLLM's continuous batching is most efficient when many requests are in-flight simultaneously, if requests across conversations are orchestrated properly

## Backend Support

**vLLM (Production)**: The primary backend, using `AsyncLLMEngine` for native async generation. vLLM handles continuous batching internally - when multiple requests are submitted concurrently, they're automatically batched for efficient GPU utilization.

**Ollama (Development)**: Provided for local development without GPU requirements. Uses async HTTP requests but lacks true GPU batching. The async framework still applies but performance gains are minimal.

## Three-Pool Architecture

The `RequestScheduler` maintains three conceptual pools that control request flow:

```text
┌──────────────────────────────────────────────────────────────────────────┐
│                              Request Flow                                │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌──────────────────┐     ┌──────────────────┐     ┌────────────────┐   │
│   │   PENDING POOL   │────▶│  PRE-DEPARTURE   │────▶│   IN-FLIGHT    │   │
│   │                  │     │      POOL        │     │                │   │
│   │  Blocked on      │     │  Ready, waiting  │     │  Executing on  │   │
│   │  dependencies    │     │  for GPU slot    │     │  GPU           │   │
│   └──────────────────┘     └──────────────────┘     └────────────────┘   │
│          │                        │                        │             │
│          │ Dependencies           │ Semaphore              │ Complete    │
│          │ satisfied              │ acquired               │             │
│          ▼                        ▼                        ▼             │
│   Check after each          Priority-based          Update state,        │
│   request completes         dispatch                check readiness      │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Pending Pool

**Definition**: Requests blocked on intra-conversation dependencies.

**Who controls it**: The `RequestScheduler` manages this pool. When a request completes, `_on_request_complete()` calls `_check_pending_for_readiness()` to move newly-ready requests to pre-departure.

**Contents**: Requests where the agent's `speak_after_within_round` list includes agents that haven't completed yet in the current round.

**Example**: Consider a 3-agent setup with participants and a moderator:

```yaml
agent_definitions:
  - agent_id: spkr_000
    role: participant
    # No speak_after_within_round → can start immediately

  - agent_id: spkr_001
    role: participant
    # No speak_after_within_round → can start immediately

  - agent_id: mod_001
    role: moderator
    role_specific_config:
      speak_after_within_round: [spkr_000, spkr_001] # must wait for both participants
```

In round 0:

- `spkr_000` and `spkr_001` requests go directly to pre-departure (no dependencies)
- `mod_001` request goes to pending pool (blocked until both participants complete)
- When both participants finish, `mod_001` moves to pre-departure

### Pre-Departure Pool

**Definition**: Ready requests waiting for GPU capacity, organized as a priority heap.

**Who controls it**: The `RequestScheduler` manages this pool. The `_scheduler_loop()` pops requests when their model's semaphore has capacity.

**Contents**: Requests with all dependencies satisfied, ordered by priority. A request stays here until its target model has a free slot on GPU.

**Data structure**: Min-heap (`heapq`) of `PrioritizedRequest` objects for O(log n) insertion and O(1) peek at highest priority.

### In-Flight Pool

**Definition**: Requests currently executing on GPU, controlled by per-model semaphores.

**Who controls it**:

- The `RequestScheduler` increments `model_in_flight[model]` before creating an async task
- The async task uses `asyncio.Semaphore` to limit concurrency
- When the task completes, it decrements `model_in_flight[model]`

**Capacity**: Each model has a semaphore sized to the effective `max_num_seqs` (min of configured upper bound and KV cache capacity). This matches vLLM's internal batching capacity.

## Decoupled Write Architecture

To maximize GPU utilization, the scheduler decouples GPU execution from file I/O:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Request Execution Flow                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   async with semaphore:                                                     │
│       result = await _execute_request()    ◀── GPU work (semaphore held)    │
│                                                                             │
│   # Semaphore RELEASED here                                                 │
│                                                                             │
│   await _on_request_complete(result)       ◀── State updates + file I/O     │
│       ├── State updates (sync, fast)                                        │
│       ├── Dependency resolution (sync)                                      │
│       └── await _finalize_conversation()   ◀── If conversation done         │
│             └── await asyncio.to_thread(progress_callback)                  │
│                   └── Blocking file I/O runs in thread pool                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why This Matters

When multiple processes write to shared files (e.g., `index.jsonl`), they contend for the same `fcntl.flock` exclusive lock. Without decoupling:

1. Process A holds semaphore, finishes GPU work, waits for file lock
2. While waiting, semaphore slot is wasted
3. Event loop is blocked by `fcntl.flock` (blocking syscall)
4. GPU starves because no new requests can be dispatched

With decoupled writes:

1. Process A finishes GPU work, **releases semaphore immediately**
2. File I/O runs in thread pool via `asyncio.to_thread()`
3. Event loop stays responsive, can dispatch new GPU requests
4. GPU stays saturated even during lock contention

### Execution Order Guarantees

The design ensures correct ordering:

1. **State updates before await**: All in-memory state changes (`conv_state`, `completed_transcripts`) happen synchronously before any `await`
2. **Dependency resolution before await**: `_check_pending_for_readiness()` and `_create_round_requests()` run before file writes
3. **Idempotent finalization**: `is_finalized` guard prevents duplicate writes even if called multiple times

### File Write Atomicity

The progress callback performs these operations:

```python
def progress_callback(completed, total, question_idx, transcript):
    # 1. Save transcript (separate file, crash-recoverable)
    transcript_manager.save_transcript(transcript)

    # 2. Atomic update under fcntl.flock:
    #    - task_manifest.json (question status → succeeded)
    #    - index.jsonl append
    bookkeeping.record_question_completion(...)
```

The `record_question_completion` holds an exclusive lock (`fcntl.flock`) and performs atomic writes (temp file + rename) to ensure `task_manifest.json` and `index.jsonl` are always consistent.

## Priority Ordering

The pre-departure pool uses a 4-tuple priority for deterministic ordering:

```python
priority = (
    0 if is_reprompt else 1,       # (1) Re-prompts first
    -rounds_completed,             # (2) More progress = higher priority
    conversation_id,               # (3) Earlier conversations (FIFO)
    round_id,                      # (4) Lower round first within conversation
)
```

### Priority Rationale

1. **Re-prompts first** (`is_reprompt`): When an agent's response fails validation (e.g., answer doesn't match choices), the retry should happen immediately to avoid blocking downstream dependencies.

1. **More progress first** (`-rounds_completed`): Conversations that have completed more rounds are closer to finishing. Prioritizing them reduces overall latency by completing conversations sooner, freeing their state memory.

1. **Earlier conversations** (`conversation_id`): FIFO ordering as a tiebreaker ensures fair scheduling and predictable behavior.

1. **Lower round first** (`round_id`): Within the same conversation, earlier rounds should complete before later rounds (though this is mostly handled by dependencies).

### Why Priority Ordering Doesn't Cause Racing

A key concern: when a GPU finishes processing a request, could reordering the queue cause race conditions?

**The answer is no**, for several reasons:

1. **Atomic state updates**: When a request completes, `_on_request_complete()` updates conversation state and resolves dependencies synchronously (before any `await`), then performs async file I/O. The scheduler loop only sees consistent state because all state mutations happen before yielding to the event loop.

1. **Semaphore-based dispatch**: Requests are only dispatched when the model has capacity. Even if priorities change, the highest-priority request with available capacity is always chosen.

1. **Per-conversation isolation**: A failed conversation only affects its own requests:

   ```python
   def _remove_conversation_requests(self, conversation_id: int) -> None:
       """Remove all pending and pre-departure requests for a failed conversation."""
       # Only removes requests matching this conversation_id
       # Other conversations are unaffected
   ```

1. **No cross-conversation dependencies**: Conversations are fully independent. Request completion in conversation X never blocks or unblocks requests in conversation Y.

## Parallelism Model

### Across Conversations

**Full parallelism**: Different conversations have zero cross-visibility. The framework processes all conversations concurrently:

```python
# All conversations initialized upfront
for idx, question in enumerate(questions):
    self._initialize_conversation(idx, question)
```

When vLLM batches requests, it doesn't matter whether they're from the same conversation or different ones - they're just independent generation requests.

### Within a Round

**Dependency-based parallelism**: Within a single round, agents respond based on dependencies, not strict serial order:

```python
# Dependencies defined per-agent in config
self.dependencies[agent_id] = set(role_config.get("speak_after_within_round", []))

def _is_agent_ready(self, agent_id: str, round_completed_agents: Set[str]) -> bool:
    """Check if agent's dependencies are satisfied."""
    deps = self.dependencies.get(agent_id, set())
    return deps.issubset(round_completed_agents)
```

**Current behavior (vanilla routing)**: With the default `VanillaRouter` and participant-only agents, all agents have empty dependencies, so all can generate concurrently within a round.

**Future extensibility**: When roles like `moderator` or `devils_advocate` are added, they can specify `speak_after_within_round` to wait for participants:

```yaml
agent_definitions:
  - agent_id: mod_001
    role: moderator
    role_specific_config:
      speak_after_within_round: [spkr_000, spkr_001, spkr_002]
```

> **Note**: `speak_after_within_round` serves dual purposes:
>
> 1. **Dependency specification** (scheduler): Determines when the agent can start generating (must wait for listed agents to complete in current round)
> 1. **Visibility specification** (prompt builder): The agent's prompt includes responses from the listed agents in the current round (in addition to the standard previous-round discussion visible to all participants)
>
> This is distinct from `visible_to` (handled by the router), which controls who can read the agent's response in subsequent rounds.

### Across Rounds

**Sequential by necessity**: Round N+1 cannot start until round N completes (agents need to see previous round's messages). However:

- Different conversations can be at different rounds simultaneously
- A conversation at round 2 can batch with another at round 0
- vLLM's continuous batching handles this transparently

## Timing and Performance

### How the Framework Speeds Things Up

1. **Continuous batching utilization**: By keeping many requests in-flight, vLLM can batch them efficiently on GPU. A single model serving 3 agents across 100 conversations can have up to 300 concurrent requests (limited by effective `max_num_seqs`).

1. **Hiding latency**: While GPU generates response for conversation A, prompts for conversations B, C, D are being prepared. This overlaps CPU work with GPU work.

1. **Optimal dispatch**: Priority ordering ensures high-value requests (re-prompts, nearly-complete conversations) get GPU time first.

### Key Configuration: `max_num_seqs_upper_bound`

This parameter sets the upper bound for maximum concurrent sequences. The effective value may be lower if limited by KV cache availability:

```yaml
model_definitions:
  llama31_8b:
    backend: vllm
    vllm_config:
      max_num_seqs_upper_bound: 256 # Upper bound; effective value limited by KV cache
```

After vLLM initializes and profiles GPU memory, the scheduler uses:

```python
effective_max_num_seqs = min(config_upper_bound, kv_cache_capacity)
self.model_semaphores[model_name] = asyncio.Semaphore(effective_max_num_seqs)
```

**How it works**:

- vLLM profiles available KV cache memory at startup
- KV cache capacity = `num_gpu_blocks * block_size / max_model_len`
- The scheduler uses `min(upper_bound, kv_cache_capacity)` for semaphore sizing
- Example: Config says 256, but KV cache only supports 110 → effective value is 110

**Trade-offs**:

- Higher upper bound = more parallelism potential (but actual value capped by hardware)
- Lower upper bound = artificially limits concurrency below hardware capacity
- Recommended: Set to 256-512 and let the system auto-calculate effective value

### Other vLLM Configuration: `attention_backend`

Some models (e.g., Gemma 2) require specific attention backends that support features like tanh softcapping:

```yaml
model_definitions:
  gemma2_27b:
    backend: vllm
    vllm_config:
      attention_backend: "FLASHINFER" # Required for Gemma 2
```

**Available backends** (depends on vLLM build):

- `FLASH_ATTN` - Default Flash Attention (FA2/FA3)
- `FLASHINFER` - FlashInfer backend
- `XFORMERS` - xFormers backend
- `TRITON_ATTN` - Triton-based attention
- `FLEX_ATTENTION` - Flex Attention

**When to use**:

- **Gemma 2 models**: Require `FLASHINFER` due to tanh softcapping
- **Most other models**: Default `FLASH_ATTN` works fine
- **Fallback**: Try `XFORMERS` if Flash Attention has issues

The backend is set via `VLLM_ATTENTION_BACKEND` environment variable internally.

## Error Handling and Cleanup

### Fatal Errors

When a conversation encounters an unrecoverable error (e.g., `MaxRetriesExceededError`):

```python
async def _on_request_complete(self, result: RequestResult) -> None:
    if not result.success:
        conv_state.error = result.error
        conv_state.is_complete = True
        # Remove queued requests for this failed conversation
        self._remove_conversation_requests(request.conversation_id)
        # Check pending pool before async finalization
        self._check_pending_for_readiness()
        # Finalize with async write (other coroutines can run during write)
        await self._finalize_conversation(conv_state, error=result.error)
        return
```

This ensures:

- Failed conversation's pending/pre-departure requests are cleaned up
- Dependency resolution happens immediately (before file writes)
- Other conversations continue unaffected
- Partial transcripts are saved for debugging

### Graceful Shutdown

The scheduler waits for all active tasks:

```python
# Wait for remaining tasks to complete
if active_tasks:
    await asyncio.gather(*active_tasks)
```

## Multi-Model Support

When agents use different models, each model gets its own:

- `AsyncLLMEngine` instance (shared by agents using that model)
- Semaphore sized to its effective `max_num_seqs`
- In-flight counter

The pre-departure pool remains unified - dispatch checks if the target model has capacity:

```python
model_name = self._get_model_for_agent(request.agent_id)
if self._has_semaphore_capacity(model_name):
    # Dispatch this request
```

This allows priority ordering across models while respecting per-model capacity limits.

## Live Status Display

Enable with `MAC_FAIRNESS_LIVE_STATUS=1` to see real-time pool status:

```text
═════════════════════════════════════════════════════════════════════════════════
 Progress: 45/100 conversations | In-Flight: 254/256 | Pre-Dep: 89 | Pending: 12
─────────────────────────────────────────────────────────────────────────────────
 PRE-DEPARTURE (ready to fly)                   │ PENDING (blocked on deps)
─────────────────────────────────────────────────────────────────────────────────
   1. [bbq_race_42] r1 spkr_000 (llama31_8b)    │   1. [bbq_race_42] r1 mod_001
   2. [bbq_race_43] r0 spkr_001 (llama31_8b)    │   2. [bbq_race_45] r2 mod_001
   ...                                          │   ...
═════════════════════════════════════════════════════════════════════════════════
```

This helps diagnose bottlenecks and verify the scheduler is behaving as expected.

> **Note**: Use `tput cnorm`, if needed, to show cursor in terminal after exp with live status concludes.
