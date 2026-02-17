"""
Live Token Usage Benchmark — TopologicPy MCP Server
====================================================
Actually calls each LLM API with the MCP tool definitions to measure
REAL token consumption. Requires API keys for cloud providers.

Environment variables needed:
    ANTHROPIC_API_KEY  — for Claude benchmarks
    OPENAI_API_KEY     — for ChatGPT benchmarks
    OLLAMA_HOST        — for Ollama (default: http://localhost:11434)

Usage:
    python benchmarks/live_benchmark.py --provider claude
    python benchmarks/live_benchmark.py --provider openai
    python benchmarks/live_benchmark.py --provider ollama --model llama3.1:8b
    python benchmarks/live_benchmark.py --all
"""

from __future__ import annotations
import argparse
import json
import time
import os
import sys
from pathlib import Path
from dataclasses import dataclass

# Import the static benchmark for tool extraction
sys.path.insert(0, str(Path(__file__).parent))
from token_benchmark import extract_tool_definitions, get_server_description


# ---------------------------------------------------------------------------
# Convert MCP tools to provider-specific format
# ---------------------------------------------------------------------------

def tools_to_anthropic_format(tools: list[dict]) -> list[dict]:
    """Convert MCP tool definitions to Anthropic's tool format."""
    return [
        {
            "name": t["name"],
            "description": t["description"],
            "input_schema": {
                "type": "object",
                "properties": t["parameters"].get("properties", {}),
            }
        }
        for t in tools
    ]


def tools_to_openai_format(tools: list[dict]) -> list[dict]:
    """Convert MCP tool definitions to OpenAI's function calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            }
        }
        for t in tools
    ]


def tools_to_ollama_format(tools: list[dict]) -> list[dict]:
    """Convert MCP tool definitions to Ollama's tool format (OpenAI-compatible)."""
    return tools_to_openai_format(tools)  # Ollama uses OpenAI format


# ---------------------------------------------------------------------------
# Test prompts (same across all providers)
# ---------------------------------------------------------------------------

TEST_PROMPTS = [
    {
        "name": "minimal",
        "prompt": "List all available topologies in the session.",
        "description": "Minimal: single tool call, small response",
    },
    {
        "name": "creation",
        "prompt": "Create a rectangular box called 'room' that is 10m wide, 15m long, and 3m tall.",
        "description": "Simple creation: one tool call with parameters",
    },
    {
        "name": "multi_step",
        "prompt": (
            "Create a 3-storey building: make 3 box cells (10x15x3m each), "
            "stack them vertically, and merge into a CellComplex called 'building'."
        ),
        "description": "Multi-step: requires planning and multiple tool calls",
    },
]


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    provider: str
    model: str
    prompt_name: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float
    tool_calls_made: int
    cost_usd: float
    error: str | None = None


def benchmark_claude(tools: list[dict], model: str = "claude-sonnet-4-5-20250929") -> list[BenchmarkResult]:
    """Run benchmark against Claude API."""
    try:
        from anthropic import Anthropic
    except ImportError:
        print("  ⚠ anthropic package not installed. Run: pip install anthropic")
        return []

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("  ⚠ ANTHROPIC_API_KEY not set")
        return []

    client = Anthropic()
    anthropic_tools = tools_to_anthropic_format(tools)
    results = []

    # Pricing (per 1K tokens)
    pricing = {
        "claude-sonnet-4-5-20250929": {"input": 0.003, "output": 0.015},
        "claude-haiku-4-5-20251001": {"input": 0.0008, "output": 0.004},
    }
    prices = pricing.get(model, pricing["claude-sonnet-4-5-20250929"])

    for test in TEST_PROMPTS:
        print(f"    Testing: {test['name']}...", end=" ", flush=True)
        start = time.time()

        try:
            response = client.messages.create(
                model=model,
                max_tokens=2048,
                system=f"You are a TopologicPy assistant. {get_server_description()}",
                tools=anthropic_tools,
                messages=[{"role": "user", "content": test["prompt"]}],
            )

            elapsed = (time.time() - start) * 1000
            usage = response.usage
            tool_calls = sum(1 for b in response.content if b.type == "tool_use")
            cost = (
                (usage.input_tokens / 1000) * prices["input"] +
                (usage.output_tokens / 1000) * prices["output"]
            )

            result = BenchmarkResult(
                provider="Claude",
                model=model,
                prompt_name=test["name"],
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.input_tokens + usage.output_tokens,
                latency_ms=elapsed,
                tool_calls_made=tool_calls,
                cost_usd=cost,
            )
            print(f"✓ {result.total_tokens} tokens, ${cost:.4f}")

        except Exception as e:
            result = BenchmarkResult(
                provider="Claude", model=model, prompt_name=test["name"],
                input_tokens=0, output_tokens=0, total_tokens=0,
                latency_ms=0, tool_calls_made=0, cost_usd=0,
                error=str(e),
            )
            print(f"✗ {e}")

        results.append(result)

    return results


def benchmark_openai(tools: list[dict], model: str = "gpt-4o") -> list[BenchmarkResult]:
    """Run benchmark against OpenAI API."""
    try:
        from openai import OpenAI
    except ImportError:
        print("  ⚠ openai package not installed. Run: pip install openai")
        return []

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("  ⚠ OPENAI_API_KEY not set")
        return []

    client = OpenAI()
    openai_tools = tools_to_openai_format(tools)
    results = []

    pricing = {
        "gpt-4o": {"input": 0.0025, "output": 0.01},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    }
    prices = pricing.get(model, pricing["gpt-4o"])

    for test in TEST_PROMPTS:
        print(f"    Testing: {test['name']}...", end=" ", flush=True)
        start = time.time()

        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=2048,
                tools=openai_tools,
                messages=[
                    {"role": "system", "content": f"You are a TopologicPy assistant. {get_server_description()}"},
                    {"role": "user", "content": test["prompt"]},
                ],
            )

            elapsed = (time.time() - start) * 1000
            usage = response.usage
            tool_calls = len(response.choices[0].message.tool_calls or [])
            cost = (
                (usage.prompt_tokens / 1000) * prices["input"] +
                (usage.completion_tokens / 1000) * prices["output"]
            )

            result = BenchmarkResult(
                provider="OpenAI",
                model=model,
                prompt_name=test["name"],
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                latency_ms=elapsed,
                tool_calls_made=tool_calls,
                cost_usd=cost,
            )
            print(f"✓ {result.total_tokens} tokens, ${cost:.4f}")

        except Exception as e:
            result = BenchmarkResult(
                provider="OpenAI", model=model, prompt_name=test["name"],
                input_tokens=0, output_tokens=0, total_tokens=0,
                latency_ms=0, tool_calls_made=0, cost_usd=0,
                error=str(e),
            )
            print(f"✗ {e}")

        results.append(result)

    return results


def benchmark_ollama(tools: list[dict], model: str = "llama3.1:8b") -> list[BenchmarkResult]:
    """Run benchmark against local Ollama instance."""
    try:
        import requests
    except ImportError:
        print("  ⚠ requests package not installed. Run: pip install requests")
        return []

    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    ollama_tools = tools_to_ollama_format(tools)
    results = []

    # Check if Ollama is running
    try:
        import requests
        resp = requests.get(f"{host}/api/tags", timeout=5)
        if resp.status_code != 200:
            print(f"  ⚠ Ollama not responding at {host}")
            return []
        # Check if model is available
        models = [m["name"] for m in resp.json().get("models", [])]
        if model not in models and f"{model}:latest" not in models:
            print(f"  ⚠ Model '{model}' not found. Available: {models}")
            print(f"    Run: ollama pull {model}")
            return []
    except Exception as e:
        print(f"  ⚠ Cannot connect to Ollama at {host}: {e}")
        return []

    for test in TEST_PROMPTS:
        print(f"    Testing: {test['name']}...", end=" ", flush=True)
        start = time.time()

        try:
            resp = requests.post(
                f"{host}/api/chat",
                json={
                    "model": model,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": f"You are a TopologicPy assistant. {get_server_description()}"},
                        {"role": "user", "content": test["prompt"]},
                    ],
                    "tools": ollama_tools,
                    "options": {"num_predict": 2048},
                },
                timeout=120,
            )

            elapsed = (time.time() - start) * 1000
            data = resp.json()

            # Ollama returns eval_count and prompt_eval_count
            input_tokens = data.get("prompt_eval_count", 0)
            output_tokens = data.get("eval_count", 0)
            tool_calls = len(data.get("message", {}).get("tool_calls", []))

            result = BenchmarkResult(
                provider="Ollama",
                model=model,
                prompt_name=test["name"],
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                latency_ms=elapsed,
                tool_calls_made=tool_calls,
                cost_usd=0.0,  # Local = free
            )
            print(f"✓ {result.total_tokens} tokens, {elapsed:.0f}ms")

        except Exception as e:
            result = BenchmarkResult(
                provider="Ollama", model=model, prompt_name=test["name"],
                input_tokens=0, output_tokens=0, total_tokens=0,
                latency_ms=0, tool_calls_made=0, cost_usd=0,
                error=str(e),
            )
            print(f"✗ {e}")

        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_live_report(all_results: list[BenchmarkResult]):
    print(f"\n{'=' * 80}")
    print("  LIVE TOKEN BENCHMARK RESULTS")
    print(f"{'=' * 80}")

    # Group by provider
    providers = {}
    for r in all_results:
        key = f"{r.provider} ({r.model})"
        providers.setdefault(key, []).append(r)

    for provider, results in providers.items():
        print(f"\n  {provider}")
        print(f"  {'─' * 70}")
        print(f"  {'Test':<20} {'Input':>8} {'Output':>8} {'Total':>8} {'Cost':>10} {'Latency':>10} {'Tools':>6}")
        print(f"  {'─' * 20} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 10} {'─' * 10} {'─' * 6}")

        for r in results:
            if r.error:
                print(f"  {r.prompt_name:<20} ERROR: {r.error}")
            else:
                cost_str = f"${r.cost_usd:.4f}" if r.cost_usd > 0 else "FREE"
                print(f"  {r.prompt_name:<20} {r.input_tokens:>8,} {r.output_tokens:>8,} "
                      f"{r.total_tokens:>8,} {cost_str:>10} {r.latency_ms:>8.0f}ms {r.tool_calls_made:>6}")

    # Cross-platform comparison
    print(f"\n{'=' * 80}")
    print("  CROSS-PLATFORM COMPARISON (multi_step scenario)")
    print(f"{'=' * 80}")
    multi_step = [r for r in all_results if r.prompt_name == "multi_step" and not r.error]
    if multi_step:
        print(f"  {'Provider':<45} {'Tokens':>8} {'Cost':>10} {'Latency':>10}")
        print(f"  {'─' * 45} {'─' * 8} {'─' * 10} {'─' * 10}")
        for r in sorted(multi_step, key=lambda x: x.total_tokens):
            cost_str = f"${r.cost_usd:.4f}" if r.cost_usd > 0 else "FREE"
            print(f"  {r.provider} ({r.model}){'':<20} {r.total_tokens:>8,} {cost_str:>10} {r.latency_ms:>8.0f}ms")
    else:
        print("  No multi_step results to compare.")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Live MCP token benchmark")
    parser.add_argument("--provider", choices=["claude", "openai", "ollama", "all"], default="all")
    parser.add_argument("--model", help="Override model name")
    args = parser.parse_args()

    print("Extracting tool definitions from server.py...")
    tools = extract_tool_definitions()
    print(f"Found {len(tools)} tools\n")

    all_results = []

    if args.provider in ("claude", "all"):
        model = args.model or "claude-sonnet-4-5-20250929"
        print(f"  Benchmarking Claude ({model})...")
        all_results.extend(benchmark_claude(tools, model))

    if args.provider in ("openai", "all"):
        model = args.model or "gpt-4o"
        print(f"\n  Benchmarking OpenAI ({model})...")
        all_results.extend(benchmark_openai(tools, model))

    if args.provider in ("ollama", "all"):
        model = args.model or "llama3.1:8b"
        print(f"\n  Benchmarking Ollama ({model})...")
        all_results.extend(benchmark_ollama(tools, model))

    if all_results:
        print_live_report(all_results)

        # Export
        output_path = Path(__file__).parent / "live_benchmark_results.json"
        with open(str(output_path), "w") as f:
            json.dump([{
                "provider": r.provider, "model": r.model,
                "prompt": r.prompt_name, "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens, "total_tokens": r.total_tokens,
                "latency_ms": r.latency_ms, "tool_calls": r.tool_calls_made,
                "cost_usd": r.cost_usd, "error": r.error,
            } for r in all_results], f, indent=2)
        print(f"Results saved to: {output_path}")
    else:
        print("\nNo results collected. Check API keys and connectivity.")


if __name__ == "__main__":
    main()
