# Reductions

Hybridizer's canonical reduction skeleton, the four ways to deliver the operator, and the OMP caveat that forces a separate body.

## Canonical skeleton (shape-B grid-strided + tree)

Identical across all four `basic-samples` reduction variants:

```csharp
var cache = new SharedMemoryAllocator<T>().allocate(blockDim.x);   // dynamic shmem
int tid = threadIdx.x + blockDim.x * blockIdx.x;
T tmp = neutral;
while (tid < N) { tmp = op(tmp, input[tid]); tid += blockDim.x * gridDim.x; }
cache[threadIdx.x] = tmp; CUDAIntrinsics.__syncthreads();

int i = blockDim.x / 2;
while (i != 0) {
    if (threadIdx.x < i) cache[threadIdx.x] = op(cache[threadIdx.x], cache[threadIdx.x + i]);
    CUDAIntrinsics.__syncthreads();
    i >>= 1;
}
if (threadIdx.x == 0) AtomicExpr.apply(ref result[0], cache[0], op);
```

Launch:

```csharp
SetDistrib(gridDimX, 1, blockDimX, 1, 1, blockDimX * sizeof(T));
```

The 6th argument is **dynamic shmem bytes** — must match what `SharedMemoryAllocator.allocate()` requests.

> ⚠ `AtomicExpr.apply` is **flagged buggy by the Hybridizer maintainer**. The variants below show all four op-delivery styles for completeness, but in new code prefer `[IntrinsicFunction("atomicAdd")]` / `("atomicMax")` stubs over a static `Atomics` helper. See [device-code.md](device-code.md) for the canonical pattern.

## The four op-delivery styles

| Sample | Op delivery | Atomic at end | Reductor kind |
|---|---|---|---|
| `1.Simple/Reduction` | Hardcoded `tmp += a[tid]` | `Interlocked.Add(ref result[0], cache[0])` | n/a (int sum only) |
| `6.Advanced/LambdaReduction` | `Func<float,float,float>` parameter, literal lambda at the `[EntryPoint]` call site | `AtomicExpr.apply(ref result[0], cache[0], reductor)` | Lambda |
| `6.Advanced/InterfacesReduction` | `ILocalReductor` instance arg; `[Kernel]` on interface + every impl; **classes** | `AtomicExpr.apply(ref result[0], cache[0], localReductor.func)` | Class via fn-ptr |
| `6.Advanced/GenericReduction` | `[HybridTemplateConcept] interface IReductor` + **struct** reductors + `[HybridRegisterTemplate(Specialize=typeof(GridReductor<XReductor>))]`; non-generic `[EntryPoint]` wrappers (generic `[EntryPoint]` not supported) | Reductor carries its own `[IntrinsicFunction("atomicAdd")]` CAS-loop stub | Struct via template |

The template path is the modern recommendation: zero-overhead dispatch, no function pointers, no `AtomicExpr.apply`.

## Two practical reduction shapes

- **Per-row independent reductions** — matvec, attention QKᵀ, RmsNorm scaling pass. `gridDim.x = rows`, one block per row, `result[blockIdx.x] = cache[0]` — atomic-free (one writer per slot). Body still uses the canonical skeleton.
- **Whole-vector reductions** — softmax-vocab max + sum-exp, RmsNorm sum-of-squares. Exactly the canonical samples: grid-strided + tree-reduce + cross-block atomic.

See [kernel-patterns.md](kernel-patterns.md) for the cooperative-block variant of the per-row case.

## OMP caveat — reductions need a separate body

In the Hybridizer OMP runtime (`hybridizer.omp.h`):

```c
inline void __syncthreads() {}                                       // no-op
#define __hybridizer_threadIdxX  omp_get_thread_num()
#define __hybridizer_blockDimX   omp_get_num_threads()
```

The canonical shape-B kernel **races** on the tree-reduce under OMP — there is no in-block barrier, because the OMP model only has one "block" (the `#pragma omp parallel` region).

**Reductions therefore need a separate OMP body**, written `Parallel.For`-style (which Hybridizer lowers to `#pragma omp parallel for`) with thread-local accumulators or `omp atomic`. Use a per-flavor `[EntryPoint]` split:

```csharp
[EntryPoint]
[HybridizerIgnore("OMP")]
public static void ReduceCuda(...) { /* canonical shape-B */ }

[EntryPoint]
[HybridizerIgnore("CUDA")]
public static void ReduceOmp(...) { /* Parallel.For + per-thread accumulator */ }
```

See [kernel-patterns.md](kernel-patterns.md) for the per-flavor split pattern in detail.

Embarrassingly-parallel kernels (SAXPY shape, `Parallel.For(0, N, i => ...)`) share *one* body across both flavors. **Reductions do not.**

## OMP wrapper replicates the whole body — beware sequential `for` over mutable state

The generated OMP wrapper for an `[EntryPoint]` is:

```c
DLL_PUBLIC int Foo_ExternCWrapper_OMP(int blockDimX, /* args */) {
    omp_set_num_threads(blockDimX);
    #pragma omp parallel
    {
        Foo(/* args */);
    }
    omp_set_num_threads(1);
    return 0;
}
```

**Every OpenMP thread runs the entire body.** Implications:

- **Read-only + write to single scalar slot** → race-but-idempotent. Fine.  
  All N threads do the same work and write the same value. Wasteful (N× work) but correct.

- **In-place mutation of an array via sequential `for`** → BROKEN.  
  Multiple threads simultaneously read/modify/write the same `x[i]`. Stomping.  
  Symptom: outputs look like `1/size`.  
  Example bug: `for (i=0; i<size; i++) { x[i] = exp(x[i]-max); sum += x[i]; }` — every thread does this, racing.

- **`Parallel.For` inside the body** → lowered to `#pragma omp parallel for` with distributed iterations. Each iteration runs exactly once across the team. Safe. Use this for any in-place mutation phase.

**Canonical OMP-safe "mutate + reduce" body** — split into a mutating `Parallel.For` followed by a sequential reduction:

```csharp
[EntryPoint]
[HybridizerIgnore("CUDA")]
public static void XxxOmp(int size, float[] x, float[] sumOut)
{
    // Phase 1: in-place mutation — distributed across the team.
    Parallel.For(0, size, i =>
    {
        x[i] = /* some computation involving x[i] */;
    });

    // Phase 2: reduction over now-stable x[] — race-but-idempotent.
    float sum = 0f;
    for (int i = 0; i < size; i++) sum += x[i];
    sumOut[0] = sum;
}
```

The `Parallel.For` has an implicit barrier at the end (OpenMP `#pragma omp for` does), so phase 2 sees a fully-written `x[]`.

**Diagnostic protocol.** When an OMP kernel produces wrong numeric output:

1. Inspect `generated-omp/<class>.omp.cpp` — find the `*_ExternCWrapper_OMP` symbol.
2. Confirm it's the `#pragma omp parallel { method(); }` shape (always is, currently).
3. Look at the underlying method body — any in-place writes outside a `Parallel.For` are suspects.

## Modern alternatives that drop the tree reduction entirely

- **Warp shuffle** — drops most `__syncthreads`. See [kernel-patterns.md](kernel-patterns.md).
- **CUB `BlockReduce`** — functionally equivalent to a hand-rolled 2-stage warp shuffle; cleaner code. See [kernel-patterns.md](kernel-patterns.md).

## Related

- Cooperative-block kernel layout: [kernel-patterns.md](kernel-patterns.md)
- Per-flavor `[EntryPoint]` split: [kernel-patterns.md](kernel-patterns.md)
- Atomic intrinsic stubs (replacement for `AtomicExpr.apply`): [device-code.md](device-code.md)
- The full gotchas digest: [gotchas.md](gotchas.md)
