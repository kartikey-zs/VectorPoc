# 🧠 Dual-Index Manager for Zero-Downtime Cleanup

## 🚨 Problem Statement

**JVector's `cleanup()` operation is NOT thread-safe** and blocks all operations:
- ❌ Cannot search during cleanup
- ❌ Cannot add vectors during cleanup
- ❌ Cannot mark deletions during cleanup

For **high-throughput production systems**, this blocking behavior is unacceptable.  
Even a **50ms cleanup** can cause a noticeable **service interruption**.

---

## ✅ Solution: Blue-Green Index Deployment

Maintain **two complete indexes** at all times:

| Role              | Purpose                                            |
|-------------------|----------------------------------------------------|
| **Active Index**  | Serves all search queries                          |
| **Standby Index** | Can be safely cleaned up without affecting queries |

---

## 🏗️ Architecture

┌─────────────────────────────────────────────────────────┐
│                   DualIndexManager                      │
└─────────────────────────────────────────────────────────┘
            │                              │
            ▼                              ▼
    ┌─────────────────┐          ┌─────────────────┐
    │   INDEX BLUE    │          │  INDEX GREEN    │
    │    (Active)     │◄────────►│   (Standby)     │
    └─────────────────┘   Swap   └─────────────────┘
            │                              │
            ▼                              ▼
        Serves queries              Can be cleaned up

---

## 🔄 Workflow

### **Phase 1: Normal Operations**
- Blue = **Active** (serving queries)
- Green = **Standby** (ready for cleanup)
- New vectors added to **both indexes**
- Deletions marked in **both indexes**

### **Phase 2: Cleanup Initiated**
- Blue continues serving queries ← **no interruption**
- Green runs `cleanup()` safely (it’s standby)
- Cleanup takes ~50ms — users don’t notice

### **Phase 3: Swap**
- Green becomes **Active** (serves queries)
- Blue becomes **Standby** (ready for cleanup)
- Atomic swap using `AtomicBoolean` — <1ms operation

### **Phase 4: Second Cleanup**
- Green now serves queries
- Blue runs `cleanup()` safely
- ✅ Both indexes are now clean and synchronized

---

## 💡 Why This Solves the Blocking Problem

1. **Zero Downtime**
    - Queries always hit the active index
    - Cleanup happens on standby only
    - Swap is atomic and instant (<1ms)

2. **Thread Safety**
    - `cleanup()` only called on inactive index
    - `addGraphNode()` is thread-safe — called on both
    - `markNodeDeleted()` is thread-safe — called on both

3. **Consistency**
    - Both indexes hold identical data
    - Additions and deletions propagate to both
    - Cleanup restores both to clean state

---

## ⚖️ Trade-offs

| Pros                            | Cons                                      |
|---------------------------------|-------------------------------------------|
| ✅ Zero downtime during cleanup  | ❌ 2× memory usage                         |
| ✅ No blocking of search queries | ❌ 2× insertion cost                       |
| ✅ Thread-safe and scalable      | ❌ Slightly more complex code              |
| ✅ Frequent cleanup possible     | ❌ Requires brief synchronization for swap |

---

## 🧭 When to Use

Use this pattern when:
- High-throughput production systems
- Cleanup downtime is **not acceptable**
- Sufficient memory available (2× overhead OK)
- Frequent cleanup needed (high deletion rates)

Avoid when:
- Memory-constrained systems
- Low query volume (occasional downtime OK)
- Proof-of-concept / dev setups (overkill)
- Cleanup can happen in maintenance windows

---

## ⚙️ Performance Characteristics

| Metric             | Description                                                                                     |
|--------------------|-------------------------------------------------------------------------------------------------|
| **Memory**         | 2× index size (e.g., 500 MB × 2 = 1 GB)                                                         |
| **Search latency** | Same as single index (only one queried)                                                         |
| **Insert latency** | ~2× single index (writes to both, can parallelize)                                              |
| **Cleanup**        | Non-blocking — standby cleanup (50 ms), swap (<1 ms), total 100 ms wall time, **0 ms downtime** |

---

**Author:** Kartikey Srivastava  
**Version:** 1.0