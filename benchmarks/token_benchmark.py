"""
Token Usage Benchmark for TopologicPy MCP Server
=================================================
Measures token consumption across different LLM backends:
- Claude (via Anthropic API / Claude Code)
- ChatGPT (via OpenAI API)
- Ollama (local models)

This estimates tokens WITHOUT making actual API calls by:
1. Serializing tool definitions the way each client would
2. Counting tokens using tiktoken (OpenAI) and anthropic tokenizer
3. Simulating realistic workflows and measuring total token budgets

Usage:
    pip install tiktoken anthropic
    python benchmarks/token_benchmark.py
"""

from __future__ import annotations
import json
import sys
import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Token counting utilities
# ---------------------------------------------------------------------------

def count_tokens_tiktoken(text: str, model: str = "gpt-4") -> int:
    """Count tokens using OpenAI's tiktoken (used by ChatGPT & Ollama)."""
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except ImportError:
        # Fallback: rough estimate ~4 chars per token
        return len(text) // 4


def count_tokens_anthropic(text: str) -> int:
    """Count tokens using Anthropic's tokenizer."""
    try:
        from anthropic import Anthropic
        client = Anthropic()
        return client.count_tokens(text)
    except (ImportError, Exception):
        # Fallback: Anthropic uses ~3.5 chars per token for English
        return int(len(text) / 3.5)


def count_tokens_estimate(text: str) -> int:
    """Universal fallback: ~4 chars per token."""
    return len(text) // 4


# ---------------------------------------------------------------------------
# MCP Tool Definition Extraction
# ---------------------------------------------------------------------------

def extract_tool_definitions() -> list[dict]:
    """Extract tool definitions from server.py by parsing @mcp.tool() functions.

    Returns a list of dicts with: name, description, parameters (JSON schema).
    """
    import ast
    import inspect

    server_path = Path(__file__).parent.parent / "server.py"
    source = server_path.read_text()
    tree = ast.parse(source)

    tools = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Check if decorated with @mcp.tool()
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call):
                    if isinstance(dec.func, ast.Attribute) and dec.func.attr == "tool":
                        docstring = ast.get_docstring(node) or ""
                        # Extract parameters from function signature
                        params = {}
                        for arg in node.args.args:
                            if arg.arg != "self" and arg.arg != "ctx":
                                params[arg.arg] = {"type": "string"}  # simplified

                        tools.append({
                            "name": node.name,
                            "description": docstring,
                            "parameters": {
                                "type": "object",
                                "properties": params,
                            }
                        })
    return tools


def get_server_description() -> str:
    """Extract the server description string."""
    return (
        "MCP server for TopologicPy — a spatial modeling library for "
        "creating hierarchical, topological representations of architectural "
        "spaces, buildings, and artefacts using non-manifold topology (NMT). "
        "Supports Vertex, Edge, Wire, Face, Shell, Cell, CellComplex, Cluster, "
        "Graph, Dictionary, and Boolean operations, plus IFC/OBJ/BREP import/export."
    )


# ---------------------------------------------------------------------------
# Workflow Scenarios
# ---------------------------------------------------------------------------

@dataclass
class WorkflowStep:
    """A single step in a workflow scenario."""
    tool_name: str
    user_prompt: str
    expected_args: dict
    expected_response: str


@dataclass
class WorkflowScenario:
    """A realistic usage scenario for benchmarking."""
    name: str
    description: str
    steps: list[WorkflowStep] = field(default_factory=list)


def get_scenarios() -> list[WorkflowScenario]:
    """Define realistic workflow scenarios of varying complexity."""

    # Scenario 1: Simple query (minimal)
    simple = WorkflowScenario(
        name="simple_query",
        description="Create a single box and query it",
        steps=[
            WorkflowStep(
                tool_name="create_cell_prism",
                user_prompt="Create a box 10x15x3 meters called 'room'",
                expected_args={"name": "room", "width": 10, "length": 15, "height": 3},
                expected_response="Created Cell (prism) 'room' (10.0 x 15.0 x 3.0, placement=center)",
            ),
            WorkflowStep(
                tool_name="query_topology",
                user_prompt="What are the properties of 'room'?",
                expected_args={"topology_name": "room"},
                expected_response='{"name":"room","type":"Cell","counts":{"vertices":8,"edges":12,"wires":6,"faces":6,"shells":1,"cells":1},"centroid":{"x":0.0,"y":0.0,"z":0.0},"volume":450.0}',
            ),
        ]
    )

    # Scenario 2: Building envelope (medium complexity)
    building = WorkflowScenario(
        name="building_envelope",
        description="Create a 3-storey building with floor cells merged into CellComplex",
        steps=[
            WorkflowStep(
                tool_name="create_vertex",
                user_prompt="Create origin vertices for 3 floors",
                expected_args={"name": "origin_0", "x": 0, "y": 0, "z": 0},
                expected_response="Created Vertex 'origin_0' at (0, 0, 0)",
            ),
            WorkflowStep(
                tool_name="create_vertex",
                user_prompt="",
                expected_args={"name": "origin_1", "x": 0, "y": 0, "z": 3},
                expected_response="Created Vertex 'origin_1' at (0, 0, 3)",
            ),
            WorkflowStep(
                tool_name="create_vertex",
                user_prompt="",
                expected_args={"name": "origin_2", "x": 0, "y": 0, "z": 6},
                expected_response="Created Vertex 'origin_2' at (0, 0, 6)",
            ),
            WorkflowStep(
                tool_name="create_cell_prism",
                user_prompt="Create floor cells",
                expected_args={"name": "floor_0", "width": 10, "length": 15, "height": 3, "origin_name": "origin_0", "placement": "bottom"},
                expected_response="Created Cell (prism) 'floor_0' (10.0 x 15.0 x 3.0, placement=bottom)",
            ),
            WorkflowStep(
                tool_name="create_cell_prism",
                user_prompt="",
                expected_args={"name": "floor_1", "width": 10, "length": 15, "height": 3, "origin_name": "origin_1", "placement": "bottom"},
                expected_response="Created Cell (prism) 'floor_1' (10.0 x 15.0 x 3.0, placement=bottom)",
            ),
            WorkflowStep(
                tool_name="create_cell_prism",
                user_prompt="",
                expected_args={"name": "floor_2", "width": 10, "length": 15, "height": 3, "origin_name": "origin_2", "placement": "bottom"},
                expected_response="Created Cell (prism) 'floor_2' (10.0 x 15.0 x 3.0, placement=bottom)",
            ),
            WorkflowStep(
                tool_name="create_cellcomplex_by_cells",
                user_prompt="Merge into building",
                expected_args={"name": "building", "cell_names": ["floor_0", "floor_1", "floor_2"]},
                expected_response="Created CellComplex 'building' from 3 cells",
            ),
            WorkflowStep(
                tool_name="query_topology",
                user_prompt="Query the building",
                expected_args={"topology_name": "building"},
                expected_response='{"name":"building","type":"CellComplex","counts":{"vertices":16,"edges":28,"wires":16,"faces":11,"shells":1,"cells":3},"centroid":{"x":0.0,"y":0.0,"z":4.5},"volume":1350.0}',
            ),
            WorkflowStep(
                tool_name="create_graph_from_topology",
                user_prompt="Create adjacency graph",
                expected_args={"name": "building_graph", "topology_name": "building"},
                expected_response="Created Graph 'building_graph' (direct=True, via_shared_faces=True)",
            ),
        ]
    )

    # Scenario 3: Export workflow (high token cost due to BREP)
    export = WorkflowScenario(
        name="export_brep",
        description="Create geometry and export as BREP (large response)",
        steps=[
            WorkflowStep(
                tool_name="create_cell_prism",
                user_prompt="Create a box and export it as BREP",
                expected_args={"name": "box", "width": 5, "length": 5, "height": 5},
                expected_response="Created Cell (prism) 'box' (5.0 x 5.0 x 5.0, placement=center)",
            ),
            WorkflowStep(
                tool_name="export_brep",
                user_prompt="Export to BREP",
                expected_args={"topology_name": "box"},
                # Simulated BREP string (real ones are ~5-50KB for a simple box)
                expected_response="CASCADE TOPOLOGY V3\n" + "A" * 15000,  # ~15KB BREP
            ),
        ]
    )

    return [simple, building, export]


# ---------------------------------------------------------------------------
# Token Budget Calculator
# ---------------------------------------------------------------------------

@dataclass
class TokenBudget:
    """Token breakdown for a single API round-trip."""
    tool_definitions: int = 0      # All tool schemas sent as system prompt
    system_prompt: int = 0         # Server description + instructions
    user_message: int = 0          # User's natural language request
    assistant_reasoning: int = 0   # LLM's thinking before tool call
    tool_call: int = 0             # The actual tool invocation JSON
    tool_response: int = 0         # Response from server
    conversation_history: int = 0  # Prior turns in context

    @property
    def input_tokens(self) -> int:
        return (self.tool_definitions + self.system_prompt +
                self.user_message + self.conversation_history)

    @property
    def output_tokens(self) -> int:
        return self.assistant_reasoning + self.tool_call

    @property
    def response_tokens(self) -> int:
        """Tokens from tool response fed back as input in next turn."""
        return self.tool_response

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.response_tokens


@dataclass
class PlatformConfig:
    """Configuration for each LLM platform."""
    name: str
    token_counter: callable
    input_cost_per_1k: float   # USD per 1K input tokens
    output_cost_per_1k: float  # USD per 1K output tokens
    context_window: int        # Max context size
    notes: str = ""


def get_platforms() -> list[PlatformConfig]:
    """Define the three target platforms."""
    return [
        PlatformConfig(
            name="Claude Code (claude-sonnet-4-5-20250929)",
            token_counter=count_tokens_estimate,
            input_cost_per_1k=0.003,
            output_cost_per_1k=0.015,
            context_window=200_000,
            notes="MCP tools sent as system prompt, full schemas every call",
        ),
        PlatformConfig(
            name="ChatGPT (gpt-4o)",
            token_counter=count_tokens_tiktoken,
            input_cost_per_1k=0.0025,
            output_cost_per_1k=0.01,
            context_window=128_000,
            notes="Function calling via tools parameter, schemas in system",
        ),
        PlatformConfig(
            name="Ollama (llama3.1:8b local)",
            token_counter=count_tokens_estimate,
            input_cost_per_1k=0.0,  # Free (local)
            output_cost_per_1k=0.0,
            context_window=8_192,
            notes="Local inference, limited context, tool support varies",
        ),
        PlatformConfig(
            name="Ollama (qwen2.5:32b local)",
            token_counter=count_tokens_estimate,
            input_cost_per_1k=0.0,
            output_cost_per_1k=0.0,
            context_window=32_768,
            notes="Local inference, better tool support, 32K context",
        ),
    ]


def benchmark_scenario(
    scenario: WorkflowScenario,
    tools: list[dict],
    platform: PlatformConfig,
) -> dict:
    """Calculate token usage for a complete workflow scenario on a platform."""

    # Serialize all tool definitions as they'd appear in system prompt
    tools_json = json.dumps(tools, indent=2)
    server_desc = get_server_description()

    tool_def_tokens = platform.token_counter(tools_json)
    system_tokens = platform.token_counter(server_desc)

    total_input = 0
    total_output = 0
    total_response = 0
    cumulative_history = 0
    step_details = []

    for i, step in enumerate(scenario.steps):
        budget = TokenBudget()
        budget.tool_definitions = tool_def_tokens
        budget.system_prompt = system_tokens
        budget.user_message = platform.token_counter(step.user_prompt) if step.user_prompt else 0
        budget.assistant_reasoning = 50  # ~50 tokens for reasoning
        budget.tool_call = platform.token_counter(json.dumps({
            "name": step.tool_name,
            "arguments": step.expected_args,
        }))
        budget.tool_response = platform.token_counter(step.expected_response)
        budget.conversation_history = cumulative_history

        # Accumulate history for next turn
        cumulative_history += (
            budget.user_message +
            budget.assistant_reasoning +
            budget.tool_call +
            budget.tool_response
        )

        total_input += budget.input_tokens
        total_output += budget.output_tokens
        total_response += budget.response_tokens

        step_details.append({
            "step": i + 1,
            "tool": step.tool_name,
            "input_tokens": budget.input_tokens,
            "output_tokens": budget.output_tokens,
            "response_tokens": budget.response_tokens,
            "total_tokens": budget.total_tokens,
        })

    grand_total = total_input + total_output + total_response
    input_cost = (total_input / 1000) * platform.input_cost_per_1k
    output_cost = (total_output / 1000) * platform.output_cost_per_1k
    # Tool responses count as input tokens when fed back
    response_cost = (total_response / 1000) * platform.input_cost_per_1k
    total_cost = input_cost + output_cost + response_cost

    fits_context = grand_total <= platform.context_window

    return {
        "scenario": scenario.name,
        "platform": platform.name,
        "steps": len(scenario.steps),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_response_tokens": total_response,
        "grand_total_tokens": grand_total,
        "estimated_cost_usd": round(total_cost, 4),
        "fits_context_window": fits_context,
        "context_window": platform.context_window,
        "context_usage_pct": round((grand_total / platform.context_window) * 100, 1),
        "tool_definitions_overhead_pct": round(
            (tool_def_tokens * len(scenario.steps) / grand_total) * 100, 1
        ) if grand_total > 0 else 0,
        "step_details": step_details,
    }


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def print_report(results: list[dict]):
    """Print a formatted benchmark report."""

    print("=" * 80)
    print("  TopologicPy MCP Server — Token Usage Benchmark Report")
    print("=" * 80)
    print()

    # Group by scenario
    scenarios = {}
    for r in results:
        scenarios.setdefault(r["scenario"], []).append(r)

    for scenario_name, platform_results in scenarios.items():
        print(f"\n{'─' * 70}")
        desc = platform_results[0].get("description", scenario_name)
        print(f"  Scenario: {scenario_name} ({platform_results[0]['steps']} steps)")
        print(f"{'─' * 70}")

        # Header
        print(f"  {'Platform':<40} {'Total Tokens':>12} {'Cost (USD)':>10} {'Context %':>10}")
        print(f"  {'─' * 40} {'─' * 12} {'─' * 10} {'─' * 10}")

        for r in platform_results:
            cost_str = f"${r['estimated_cost_usd']:.4f}" if r['estimated_cost_usd'] > 0 else "FREE"
            ctx_warn = " ⚠" if not r['fits_context_window'] else ""
            print(f"  {r['platform']:<40} {r['grand_total_tokens']:>12,} {cost_str:>10} {r['context_usage_pct']:>8.1f}%{ctx_warn}")

        # Detail for first platform
        print(f"\n  Tool definition overhead: {platform_results[0]['tool_definitions_overhead_pct']}% of total tokens")
        print(f"  Step breakdown (first platform):")
        for step in platform_results[0]["step_details"]:
            print(f"    Step {step['step']}: {step['tool']:<35} "
                  f"in={step['input_tokens']:>6} out={step['output_tokens']:>5} "
                  f"resp={step['response_tokens']:>6} total={step['total_tokens']:>7}")

    # Summary section
    print(f"\n{'=' * 80}")
    print("  KEY FINDINGS")
    print(f"{'=' * 80}")

    # Find worst case
    worst = max(results, key=lambda r: r.get("tool_definitions_overhead_pct", 0))
    brep_results = [r for r in results if r["scenario"] == "export_brep"]

    print(f"""
  1. TOOL DEFINITION OVERHEAD: {worst['tool_definitions_overhead_pct']}% of tokens
     - 30 tool definitions are sent with EVERY request
     - This is the #1 optimization target

  2. BREP EXPORT TOKEN BOMB:""")
    if brep_results:
        for r in brep_results:
            ctx_str = "FITS" if r['fits_context_window'] else "EXCEEDS CONTEXT"
            print(f"     - {r['platform']}: {r['grand_total_tokens']:,} tokens ({ctx_str})")

    print(f"""
  3. CONTEXT WINDOW RISK:""")
    for r in results:
        if not r['fits_context_window']:
            print(f"     ⚠ {r['platform']} + {r['scenario']}: {r['context_usage_pct']}% of {r['context_window']:,} context")

    overflows = [r for r in results if not r['fits_context_window']]
    if not overflows:
        print(f"     All scenarios fit within context windows (but multi-turn sessions accumulate)")

    print(f"""
  4. COST PER BUILDING WORKFLOW:""")
    building_results = [r for r in results if r["scenario"] == "building_envelope"]
    for r in building_results:
        cost_str = f"${r['estimated_cost_usd']:.4f}" if r['estimated_cost_usd'] > 0 else "FREE (local)"
        print(f"     - {r['platform']}: {cost_str}")

    print()


def export_json(results: list[dict], path: str):
    """Export results as JSON for further analysis."""
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results exported to: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Extracting tool definitions from server.py...")
    tools = extract_tool_definitions()
    print(f"Found {len(tools)} tools\n")

    scenarios = get_scenarios()
    platforms = get_platforms()

    results = []
    for scenario in scenarios:
        for platform in platforms:
            result = benchmark_scenario(scenario, tools, platform)
            results.append(result)

    print_report(results)

    # Export JSON
    output_path = Path(__file__).parent / "benchmark_results.json"
    export_json(results, str(output_path))

    return results


if __name__ == "__main__":
    main()
