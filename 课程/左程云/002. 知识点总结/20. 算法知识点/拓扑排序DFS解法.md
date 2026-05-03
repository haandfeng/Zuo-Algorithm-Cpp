# 拓扑排序DFS解法
#拓扑排序 

---

拓扑排序的另一个标准/定义:

拓扑排序中最末尾的节点(别人都做好了才能完成的节点), 这个节点孩子为0, 他自己做完的时间为1
同样道理, 如果一个节点他做完的时间越大, 他子节点遍历的东西越多, 最后完成的时间越晚, 这种节点在拓扑排序中需要排在前面.

代码中的out的含义:
如果反过来看, 一个节点, 他所有孩子都处理完成了, 他才能搞定, 即他遍历完所有孩子才能轮到他, 他遍历的时间越晚, 他拓扑排序就越往前, 如果一个末尾节点没有什么孩子, 他完成的时间早, 那么他拓扑排序排后面

out也可以理解为完成的时间


```java
// https://www.lintcode.com/problem/topological-sorting/
public class DFS {

	// 不要提交这个类
	public static class DirectedGraphNode {
		public int label;
		public ArrayList<DirectedGraphNode> neighbors;

		public DirectedGraphNode(int x) {
			label = x;
			neighbors = new ArrayList<DirectedGraphNode>();
		}
	}

	// 提交下面的
	public static class Record {
		public DirectedGraphNode node;
		public long out;

		public Record(DirectedGraphNode n, long o) {
			node = n;
			out = o;
		}
	}

	public static class MyComparator implements Comparator<Record> {

		@Override
		public int compare(Record o1, Record o2) {
			if (o2.out > o1.out) {
				return 1;
			} else if (o1.out > o2.out) {
				return -1;
			} else {
				return 0;
			}
		}
	}

	public static ArrayList<DirectedGraphNode> topSort(ArrayList<DirectedGraphNode> graph) {
		HashMap<DirectedGraphNode, Record> order = new HashMap<>();
		for (DirectedGraphNode cur : graph) {
			f(cur, order);
		}
		ArrayList<Record> recordArr = new ArrayList<>();
		for (Record r : order.values()) {
			recordArr.add(r);
		}
		recordArr.sort(new MyComparator());
		ArrayList<DirectedGraphNode> ans = new ArrayList<DirectedGraphNode>();
		for (Record r : recordArr) {
			ans.add(r.node);
		}
		return ans;
	}

	public static Record f(DirectedGraphNode cur, HashMap<DirectedGraphNode, Record> order) {
		if (order.containsKey(cur)) {
			return order.get(cur);
		}
		long out = 1;
		for (DirectedGraphNode next : cur.neighbors) {
			out += f(next, order).out;
		}
		Record ans = new Record(cur, out);
		order.put(cur, ans);
		return ans;
	}

}


```
