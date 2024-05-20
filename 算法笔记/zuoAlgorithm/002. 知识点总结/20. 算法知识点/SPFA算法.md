# SPFA算法
#单源最短路径算法 

---

SPFA (shortest path faster algorithm) 是一个[[单源最短路径算法|单源最短路径算法]]，与另一个单源最短路算法[[Dijkstra算法]]
不同的是SPFA可以用来处理含有负权的图，并且也可以判断图中是否存在负权回路。


实际上，SPFA是[[Bellman-Ford算法]]的一种队列实现，减少了不必要的冗余计算。SPFA的时间复杂度可以达到O(kE)，k是每一个节点的平均进队次数，可以证明k不大于2。