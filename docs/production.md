# Production safety

munkres is built so that a problem in its optional native accelerator can never break the program using it.
The pure-Python solver is always present, exact, and gives the same answers as the native kernel.

## What happens when something goes wrong

| Situation | Behaviour |
| --- | --- |
| C extension not built, wrong ABI, or fails to import | `import munkres` still works; pure Python is used. Reason in `native_status()["disabled_reason"]`. |
| Extension loads but returns wrong answers (miscompiled, corrupted) | A self-test runs once at load; a failing kernel is rejected before it is ever used. |
| The kernel raises an unexpected exception during a call | The exception is logged once per call site, that call finishes on the exact pure-Python solver, and the caller gets a correct result. |
| Repeated kernel failures | After 3 failures the **circuit breaker** switches the kernel off for the process (no more time wasted on a broken path). One failing call counts as one failure. |
| Inputs the kernel cannot represent exactly (huge ints, extended precision, Fractions) | Routed to the exact solver; results are never silently rounded. |
| Genuinely bad input (infeasible problem, NaN) | Raised as `UnsolvableMatrix` / `ValueError`, exactly as without the kernel. These are never mistaken for kernel faults. |

Nothing is ever printed; diagnostics go to the standard `logging` logger named `munkres`.

## Monitoring

```python
import munkres
munkres.native_status()
# {'available': True, 'version': 'c-kernel-abi2', 'disabled_reason': None, 'failures': 0,
#  'recent_failures': [], 'breaker_limit': 3, 'strict': False}
```

Export `failures` and `available` to your metrics system, and alert when `available` is false in an
environment that should have the kernel. `munkres.reset_native()` re-arms the kernel after a fix.

## Switches

| Switch | Effect |
| --- | --- |
| `PALDITA_NO_NATIVE=1` | Never use the native kernel (pure Python). |
| `PALDITA_STRICT=1` / `munkres.set_strict_mode(True)` | Raise on unexpected native failures. Use in CI so faults are not hidden. |

## Guarantees and limits

- Results from the fallback path are identical in total cost to the native path (tested by fault injection).
- When several optimal assignments exist the one returned may differ between paths; the total cost never does.
- Thread and async safe: the per-call fallback state is context-local; counters are lock-protected.
- Pure Python is slower than the kernel (see [performance](performance.md)); a tripped breaker trades speed for safety.
