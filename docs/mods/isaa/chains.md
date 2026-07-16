# Chains

###### PIPELINE DSL · VERIFIED AGAINST `31a117e`

`base/Agent/chain.py` implements composable task pipelines over agents, functions, and other chains — wired together with Python operators.

## Operator DSL

| Operator | Produces | Meaning |
|---|---|---|
| `a >> b` | `Chain` | Sequential: run `a`, feed result into `b` |
| `a + b` | `ParallelChain` | Run `a` and `b` in parallel |
| `a & b` | `ParallelChain` | Alias for `+` |
| `a | b` | `ErrorHandlingChain` | If `a` raises, run fallback `b` |
| `cond % b` | `ConditionalChain` | Else-branch: used after a condition to define the false path |

```python
pipeline = fetch >> parse >> (summarize + tag) >> store | alert_on_failure
result = await pipeline.run(input_data)
```

## Building blocks

| Class | Role |
|---|---|
| `Chain` | Main sequential chain — created by `>>` |
| `ParallelChain` | Parallel execution of agents/chains |
| `ConditionalChain` | Branching on a condition |
| `ErrorHandlingChain` | Primary chain with fallback path |
| `Function` | Wraps a native Python function as a chainable component |
| `CF` | Chain Format — formatting/data extraction between tasks |
| `IS` | Conditional check used for branching logic |
| `ChainRunType` | Enum of run modes |

## Stored chains (DSL strings)

Chains can be persisted and executed by name — this powers icli's `/chain` commands (e.g. `autodoc_guided`). The stored form is a DSL string of `tool:` calls and `@self("prompt")` blocks joined with `>>`; see [icli](../../flows/icli.md) for the runner.

<!-- verified: toolboxv2/mods/isaa/base/Agent/chain.py::ChainBase.__rshift__,__add__,__and__,__or__,__mod__ + class docstrings @ 31a117e -->
