# 数据结构设计之O(1)实现setAll

---

数据结构设计题，遇到了千万别慌，为啥不会让你用复杂结构设计的，他一定是我在体系班里讲过的，
结构都不会到线段树这么难，或者是什么哈希表双链表这样结构拼出来的，所以不用慌，
你就你就认为这个东西能用简单结构拼出来，而且他就是只用简单结构，连线段树这样的复杂结构都到不了啊

```java
public static HashMap<Integer, int[]> map = new HashMap<>();  
public static int setAllValue;  
public static int setAllTime;  
public static int cnt;  
  
public static void put(int k, int v) {  
   if (map.containsKey(k)) {  
      int[] value = map.get(k);  
      value[0] = v;  
      value[1] = cnt++;  
   } else {  
      map.put(k, new int[] { v, cnt++ });  
   }  
}  
  
public static void setAll(int v) {  
   setAllValue = v;  
   setAllTime = cnt++;  
}  
  
public static int get(int k) {  
   if (!map.containsKey(k)) {  
      return -1;  
   }  
   int[] value = map.get(k);  
   if (value[1] > setAllTime) {  
      return value[0];  
   } else {  
      return setAllValue;  
   }  
}  
  
public static int n, op, a, b;  
  
public static void main(String[] args) throws IOException {  
   BufferedReader br = new BufferedReader(new InputStreamReader(System.in));  
   StreamTokenizer in = new StreamTokenizer(br);  
   PrintWriter out = new PrintWriter(new OutputStreamWriter(System.out));  
   while (in.nextToken() != StreamTokenizer.TT_EOF) {  
      map.clear();  
      setAllValue = 0;  
      setAllTime = -1;  
      cnt = 0;  
      n = (int) in.nval;  
      for (int i = 0; i < n; i++) {  
         in.nextToken();  
         op = (int) in.nval;  
         if (op == 1) {  
            in.nextToken();  
            a = (int) in.nval;  
            in.nextToken();  
            b = (int) in.nval;  
            put(a, b);  
         } else if (op == 2) {  
            in.nextToken();  
            a = (int) in.nval;  
            out.println(get(a));  
         } else {  
            in.nextToken();  
            a = (int) in.nval;  
            setAll(a);  
         }  
      }  
   }  
   out.flush();  
   out.close();  
   br.close();  
}
```