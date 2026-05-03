# Kruskal算法代码

#Kruskal 

---


```java
public static class EdgeComparator implements Comparator<Edge> {

    @Override
    public int compare(Edge o1, Edge o2) {
        return o1.weight - o2.weight;
    }

}

public static Set<Edge> kruskalMST(Graph graph) {
    UnionFind unionFind = new UnionFind();
    unionFind.makeSets(graph.nodes.values());
    PriorityQueue<Edge> priorityQueue = new PriorityQueue<>(new EdgeComparator());
    for (Edge edge : graph.edges) { // M 条边
        priorityQueue.add(edge);  // O(logM)
    }
    Set<Edge> result = new HashSet<>();
    while (!priorityQueue.isEmpty()) { // M 条边
        Edge edge = priorityQueue.poll(); // O(logM)
        if (!unionFind.isSameSet(edge.from, edge.to)) { // O(1)
            result.add(edge);
            unionFind.union(edge.from, edge.to);
        }
    }
    return result;
}

```