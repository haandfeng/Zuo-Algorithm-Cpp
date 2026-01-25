```java
import java.util.*;
import java.util.function.*;

public class Result {

    /** Active connection record. target is 1-based index of the server. */
    static class Conn {
        String connId, userId, objId;
        int target; // 1-based
        Conn(String c, String u, String o, int t) {
            connId = c; userId = u; objId = o; target = t;
        }
    }

    /**
     * Simulates the load balancer routing logic.
     *
     * Returns a log entry for every successful connect (initial or reroute):
     * "connectionId,userId,targetIndex"
     */
    public static List<String> route_request(int numTargets,
                                             int maxConnectionsPerTarget,
                                             List<String> requests) {
        List<String> log = new ArrayList<>();

        // Treat very large value as "no limit" (optional convenience).
        // If caller passes something huge already, this is harmless.
        int cap = maxConnectionsPerTarget <= 0 ? Integer.MAX_VALUE : maxConnectionsPerTarget;

        // Active connection count per target (1..numTargets).
        int[] load = new int[numTargets + 1];

        // Availability: false only during a SHUTDOWN handling.
        boolean[] available = new boolean[numTargets + 1];
        Arrays.fill(available, true);

        // Lookup active connection by connectionId.
        Map<String, Conn> byConnId = new HashMap<>();

        // Object affinity: objId -> pinned target (only while object has active conns)
        Map<String, Integer> objToTarget = new HashMap<>();
        Map<String, Integer> objCount = new HashMap<>();

        // Per-target active connections in ascending connectionId order (for SHUTDOWN eviction).
        Map<Integer, TreeMap<String, Conn>> targetConns = new HashMap<>();
        for (int t = 1; t <= numTargets; t++) {
            targetConns.put(t, new TreeMap<>());
        }

        // Choose a target for a connection under current state.
        // 1) If object is pinned: only that target is allowed, else reject.
        // 2) Otherwise pick least-loaded among eligible targets; tie by smaller index.
        Function<Conn, Integer> choose = (Conn c) -> {
            Integer pinned = objToTarget.get(c.objId);
            if (pinned != null) {
                if (available[pinned] && load[pinned] < cap) {
                    return pinned;
                }
                return -1; // pinned but not eligible => reject
            }
            int best = -1;
            int bestLoad = Integer.MAX_VALUE;
            for (int t = 1; t <= numTargets; t++) {
                if (!available[t]) {
                    continue;
                }
                if (load[t] >= cap) {
                    continue;
                }
                if (best == -1 || load[t] < bestLoad || (load[t] == bestLoad && t < best)) {
                    best = t;
                    bestLoad = load[t];
                }
            }
            return best; // -1 => reject
        };

        // Attach connection to target (updates all indexes and logs).
        BiConsumer<Conn, Integer> attach = (Conn c, Integer t) -> {
            c.target = t;
            byConnId.put(c.connId, c);
            targetConns.get(t).put(c.connId, c);
            load[t]++;

            // Set/maintain object affinity
            objToTarget.putIfAbsent(c.objId, t);
            objCount.put(c.objId, objCount.getOrDefault(c.objId, 0) + 1);

            log.add(c.connId + "," + c.userId + "," + t);
        };

        // Detach connection from its current target (no log for disconnect).
        Consumer<Conn> detach = (Conn c) -> {
            int t = c.target;
            if (targetConns.get(t).remove(c.connId) != null) {
                load[t]--;
            }
            byConnId.remove(c.connId);

            int left = objCount.getOrDefault(c.objId, 0) - 1;
            if (left <= 0) {
                objCount.remove(c.objId);
                objToTarget.remove(c.objId);
            } else {
                objCount.put(c.objId, left);
            }
        };

        // Process requests in order
        for (String raw : requests) {
            String[] parts = Arrays.stream(raw.split(","))
                    .map(String::trim)
                    .toArray(String[]::new);
            String action = parts[0];

            if ("CONNECT".equals(action)) {
                // CONNECT, connectionId, userId, objectId
                Conn c = new Conn(parts[1], parts[2], parts[3], -1);
                int t = choose.apply(c);
                if (t != -1) {
                    attach.accept(c, t);
                } // reject => no log
            } else if ("DISCONNECT".equals(action)) {
                // DISCONNECT, connectionId, userId, objectId
                Conn c = byConnId.get(parts[1]);
                if (c != null) {
                    detach.accept(c);
                } // assumed valid & active
            } else if ("SHUTDOWN".equals(action)) {
                // SHUTDOWN, targetIndex
                int down = Integer.parseInt(parts[1]);

                // 1) Mark target unavailable
                available[down] = false;

                // 2) Evict all active connections on that target in connId ascending order
                List<Conn> evicted = new ArrayList<>(targetConns.get(down).values());

                // 3) Detach first to clear loads and affinities
                for (Conn c : evicted) {
                    detach.accept(c);
                }

                // 4) Reroute evicted connections in connId order with same rules (down target not eligible)
                for (Conn c : evicted) {
                    c.target = -1; // reset (not strictly required, but clearer)
                    int t = choose.apply(c);
                    if (t != -1) {
                        attach.accept(c, t);
                    } // only successful reroutes are logged
                }

                // 5) Target becomes available again after shutdown handling
                available[down] = true;
            }
        }

        return log;
    }
}
```

# Stripe “route_requests” — Load-balancing WebSocket Connections

Implement a function to simulate a load balancer that routes longlived websocket **CONNECT** requests to one of several Jupyter target servers. You must also handle **DISCONNECT** events, perserver **capacity limits**, and **SHUTDOWN** events that temporarily evict and reroute existing connections.

---

## Function Signature (Java)

```java
List<String> route_request(
int numTargets,
int maxConnectionsPerTarget,
List<String> requests
)
```
- `numTargets`: number of target Jupyter servers. Targets are **1 based**.
- `maxConnectionsPerTarget`: perserver capacity. (Used in Part 4. If very large, assume “no limit”.)
- `requests`: sequence of input lines in order received.

---

## Request Line Schemas

1. `action,connectionId,userId,objectId` — used for `CONNECT` and `DISCONNECT`
2. `action,targetIndex` — used for `SHUTDOWN` (with 1 based `targetIndex`)

- `action ∈ { CONNECT, DISCONNECT, SHUTDOWN }`

---

## Return Value (Log)

Return a list of strings, **one for every successful connection** (initial routes and any reroutes after shutdowns), formatted as:

```
connectionId,userId,targetIndex
```

Rejected connections are **not** logged.

---

## Rules by Part

### Part 1 — Basic Load Balancing (tests 1–4)
- On each `CONNECT`, choose the target with the **fewest active connections**.
- Break ties by **smaller target index**.
- Routing is based only on the events seen so far (order matters).

### Part 2 — Disconnections (tests 5–8)
- On `DISCONNECT, connectionId, ...`, remove that active connection from its target.
(You may assume the `connectionId` is valid and active.)

### Part 3 — Object Affinity (tests 9–11)
- If any active connection exists for an `objectId`, **all new connections** with that same `objectId` **must connect to the same target** (the “pinned” server), even if that is not the leastloaded option.
- If no active connection exists for the object, route normally per Part 1.
- The user ID does **not** affect routing—only the object ID creates affinity.
- When the last connection for an object ends, the affinity disappears.

### Part 4 — Target Capacity (tests 12–14)
- Each target has a max of `maxConnectionsPerTarget` active connections.
- When routing, **exclude** targets that are **at capacity**.
- If an object is pinned to a target that is **full**, the new `CONNECT` is **rejected** (no log).
- If all eligible targets are full, the `CONNECT` is **rejected**.

### Part 5 — Target Shutdown (tests 15–17)
- `SHUTDOWN,targetIndex` makes that target **temporarily unavailable** and **evicts all its active connections**.
- Evicted connections must be **rerouted in ascending `connectionId` order** using the same rules (least load → tie by index; object affinity; capacity). The shuttingdown target is **not eligible** while down.
- **Log** each successful reroute.
- After processing evictions, the target is considered **available again** for future routing.

---

## Examples

### Part 1
**Targets:** 2
**Requests:**
```
CONNECT,conn1,userA,obj1
CONNECT,conn2,userB,obj2
```
**Output:**
```
conn1,userA,1
conn2,userB,2
```

### Part 2
**Targets:** 2
**Requests:**
```
CONNECT,conn1,userA,obj1
DISCONNECT,conn1,userA,obj1
CONNECT,conn2,userB,obj2
```
**Output:**
```
conn1,userA,1
conn2,userB,1
```

### Part 3 — Affinity
**Example 1**
```
CONNECT,conn1,userA,obj1
CONNECT,conn2,userB,obj1
```
**Output**
```
conn1,userA,1
conn2,userB,1
```

**Example 2**
```
CONNECT,conn1,userA,obj1
DISCONNECT,conn1,userA,obj1
CONNECT,conn2,userB,obj2
CONNECT,conn3,userA,obj1
```
**Output**
```
conn1,userA,1
conn2,userB,1
conn3,userA,2
```

### Part 4 — Capacity
**Targets:** 2, `maxConnectionsPerTarget = 1`
**Requests:**
```
CONNECT,conn1,userA,obj1
CONNECT,conn2,userB,obj2
CONNECT,conn3,userC,obj3
```
**Output:**
```
conn1,userA,1
conn2,userB,2
```
(`conn3` is rejected and not logged.)

### Part 5 — Shutdown & Reroute
**Targets:** 2, large capacity
**Requests:**
```
CONNECT,conn1,userA,obj1
CONNECT,conn2,userB,obj2
SHUTDOWN,1
CONNECT,conn3,userC,obj3
```
**Output:**
```
conn1,userA,1
conn2,userB,2
conn1,userA,2 <-- rerouted during shutdown
conn3,userC,1
```

---

## Clarifications
- All target indices are **1based** across the problem.
- Only **successful** connects (including reroutes) are logged.
- During `SHUTDOWN`, remove/evict connections *first* (which may clear some object affinities), then reroute evictees in `connectionId` order using the same rules as normal connects.

---

## Reference Java Implementation (with inline comments)

```java
import java.util.*;

class Result {

/**
* Connection record kept while the connection is active.
* target is 1-based index of the server this connection is attached to.
*/
static class Conn {
String connId, userId, objId;
int target; // 1-based
Conn(String c, String u, String o, int t) {
connId = c; userId = u; objId = o; target = t;
}
}

/**
* Simulates the load balancer routing logic.
*
* Returns a log entry for every successful connect (initial or re-route):
* "connectionId,userId,targetIndex"
*/
public static List<String> route_request(int numTargets,
int maxConnectionsPerTarget,
List<String> requests) {
List<String> log = new ArrayList<>();

// Active connection count per target (index 1..numTargets).
int[] load = new int[numTargets + 1];

// Whether each target is currently eligible to receive connections.
// It becomes false during a SHUTDOWN, then back to true after re-routing.
boolean[] available = new boolean[numTargets + 1];
Arrays.fill(available, true);

// Lookup an active connection by connectionId.
Map<String, Conn> byConnId = new HashMap<>();

// Object affinity:
// If an object has an active connection, objToTarget[objId] = pinned target.
// We also keep how many active connections use the object, so when it drops to 0
// we can remove the affinity.
Map<String, Integer> objToTarget = new HashMap<>();
Map<String, Integer> objCount = new HashMap<>();

// For SHUTDOWN eviction we must re-route evicted connections in connectionId order.
// Keep per-target connections in a sorted map keyed by connectionId.
Map<Integer, TreeMap<String, Conn>> targetConns = new HashMap<>();
for (int t = 1; t <= numTargets; t++) targetConns.put(t, new TreeMap<>());

// -------------------------------------------------- Helper lambdas --------------------------------------------------

// Choose a target for a connection under current state.
// 1) If object affinity exists, try to use that target (if available & not full).
// 2) Otherwise pick least-loaded; tie-break by smaller index.
java.util.function.Function<Conn, Integer> choose = (Conn c) -> {
Integer pinned = objToTarget.get(c.objId);
if (pinned != null) {
return (available[pinned] && load[pinned] < maxConnectionsPerTarget) ? pinned : -1;
}
int best = -1, bestLoad = Integer.MAX_VALUE;
for (int t = 1; t <= numTargets; t++) {
if (!available[t]) continue; // skip down target
if (load[t] >= maxConnectionsPerTarget) continue; // skip full target
if (best == -1 || load[t] < bestLoad || (load[t] == bestLoad && t < best)) {
best = t; bestLoad = load[t];
}
}
return best; // -1 means reject
};

// Attach connection to target: update all indexes and write to log.
java.util.function.BiConsumer<Conn, Integer> attach = (Conn c, Integer t) -> {
c.target = t;
byConnId.put(c.connId, c);
targetConns.get(t).put(c.connId, c);
load[t]++;
// Establish object affinity if it's the first active conn for that object.
objToTarget.putIfAbsent(c.objId, t);
objCount.put(c.objId, objCount.getOrDefault(c.objId, 0) + 1);
log.add(c.connId + "," + c.userId + "," + t);
};

// Detach connection from its current target (no log for disconnect).
java.util.function.Consumer<Conn> detach = (Conn c) -> {
int t = c.target;
if (targetConns.get(t).remove(c.connId) != null) {
load[t]--;
}
byConnId.remove(c.connId);
// Update object affinity reference counts.
int left = objCount.getOrDefault(c.objId, 0) - 1;
if (left <= 0) {
objCount.remove(c.objId);
objToTarget.remove(c.objId);
} else {
objCount.put(c.objId, left);
}
};

// -------------------------------------------------- Process each request in order --------------------------------------------------
for (String raw : requests) {
// Split and trim to tolerate spaces in the input.
String[] parts = Arrays.stream(raw.split(",")).map(String::trim).toArray(String[]::new);
String action = parts[0];

if ("CONNECT".equals(action)) {
// CONNECT, connectionId, userId, objectId
Conn c = new Conn(parts[1], parts[2], parts[3], -1);
int t = choose.apply(c);
if (t != -1) attach.accept(c, t); // rejected connects are not logged
}
else if ("DISCONNECT".equals(action)) {
// DISCONNECT, connectionId, userId, objectId
Conn c = byConnId.get(parts[1]);
if (c != null) detach.accept(c); // guaranteed valid per prompt
}
else if ("SHUTDOWN".equals(action)) {
// SHUTDOWN, targetIndex
int down = Integer.parseInt(parts[1]);

// 1) Mark as unavailable so it's excluded from routing decisions.
available[down] = false;

// 2) Evict all connections from this target in connectionId order.
List<Conn> evicted = new ArrayList<>(targetConns.get(down).values());

// 3) Remove them first to clear loads / object affinities.
for (Conn c : evicted) detach.accept(c);

// 4) Re-route evicted connections using the same rules.
for (Conn c : evicted) {
int t = choose.apply(c);
if (t != -1) attach.accept(c, t); // only successful re-routes are logged
}

// 5) Target immediately becomes available again after shutdown completes.
available[down] = true;
}
}

return log;
}
}
```