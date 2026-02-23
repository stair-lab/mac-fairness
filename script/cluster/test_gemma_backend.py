"""Test vLLM attention backends for gemma-2-27b on B200.

Usage:
    CUDA_VISIBLE_DEVICES=4 TEST_BACKEND=TRITON_ATTN python script/cluster/test_gemma_backend.py
    CUDA_VISIBLE_DEVICES=4 TEST_BACKEND=FLASHINFER python script/cluster/test_gemma_backend.py
    CUDA_VISIBLE_DEVICES=4 TEST_BACKEND=FLASH_ATTN python script/cluster/test_gemma_backend.py
    CUDA_VISIBLE_DEVICES=4 TEST_BACKEND=AUTO python script/cluster/test_gemma_backend.py
"""
import os
import sys
import time
import signal


def test_backend(backend_name, timeout_sec=180):
    from vllm import AsyncLLMEngine, AsyncEngineArgs

    kwargs = dict(
        model="google/gemma-2-27b-it",
        gpu_memory_utilization=0.9,
        max_model_len=2048,
        enforce_eager=True,
        tensor_parallel_size=1,
    )
    if backend_name != "AUTO":
        kwargs["attention_backend"] = backend_name

    start = time.strftime("%H:%M:%S")
    print(f"[{start}] Testing backend: {backend_name} (timeout={timeout_sec}s)", flush=True)

    def timeout_handler(signum, frame):
        raise TimeoutError(f"Backend {backend_name} timed out after {timeout_sec}s")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_sec)
    try:
        engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**kwargs))
        signal.alarm(0)
        end = time.strftime("%H:%M:%S")
        print(f"[{end}] SUCCESS with {backend_name}", flush=True)
        return True
    except TimeoutError as e:
        end = time.strftime("%H:%M:%S")
        print(f"[{end}] TIMEOUT: {e}", flush=True)
        return False
    except Exception as e:
        signal.alarm(0)
        end = time.strftime("%H:%M:%S")
        print(f"[{end}] ERROR with {backend_name}: {type(e).__name__}: {e}", flush=True)
        return False


if __name__ == "__main__":
    backend = os.environ.get("TEST_BACKEND", "TRITON_ATTN")
    timeout = int(os.environ.get("TEST_TIMEOUT", "180"))
    ok = test_backend(backend, timeout_sec=timeout)
    sys.exit(0 if ok else 1)
