# Java 快速IO

---

https://oi-wiki.org/lang/java-pro/

更高速的输入输出
Scanner 和 System.out.print 在最开始会工作得很好，但是在处理更大的输入的时候会降低效率，因此我们会需要使用一些方法来提高 IO 速度。

### 使用 Kattio + StringTokenizer

```java
class Kattio extends PrintWriter {
    private BufferedReader r;
    private StringTokenizer st;
    // 标准 IO
    public Kattio() { this(System.in, System.out); }
    public Kattio(InputStream i, OutputStream o) {
        super(o);
        r = new BufferedReader(new InputStreamReader(i));
    }
    // 文件 IO
    public Kattio(String intput, String output) throws IOException {
        super(output);
        r = new BufferedReader(new FileReader(intput));
    }
    // 在没有其他输入时返回 null
    public String next() {
        try {
            while (st == null || !st.hasMoreTokens())
                st = new StringTokenizer(r.readLine());
            return st.nextToken();
        } catch (Exception e) {}
        return null;
    }
    public int nextInt() { return Integer.parseInt(next()); }
    public double nextDouble() { return Double.parseDouble(next()); }
    public long nextLong() { return Long.parseLong(next()); }
}


class Test {
    public static void main(String[] args) {
        Kattio io = new Kattio();
        // 字符串输入
        String str = io.next();
        // int 输入
        int num = io.nextInt();
        // 输出
        io.println("Result");
        // 请确保关闭 IO 流以确保输出被正确写入
        io.close();
    }
}
```


### 使用 StreamTokenizer 作为输入

```java
import java.io.*;
public class Main {
    // IO 代码
    public static StreamTokenizer in = new StreamTokenizer(new BufferedReader(new InputStreamReader(System.in), 32768));
    public static PrintWriter out = new PrintWriter(new OutputStreamWriter(System.out));
    public static double nextDouble() throws IOException { in.nextToken(); return in.nval; }
    public static float nextFloat() throws IOException { in.nextToken(); return (float)in.nval; }
    public static int nextInt() throws IOException { in.nextToken(); return (int)in.nval; }
    public static String next() throws IOException { return in.sval; }
    public static long nextLong() throws Exception { in.nextToken(); return (long)in.nval;}
    
    // 使用示例
    public static void main(String[] args) throws Exception {
        int n = nextInt();
        out.println(n);
        out.close();
    }
}

```


例子: 

```java

public static void main(String[] args) throws IOException {  
   BufferedReader br = new BufferedReader(new InputStreamReader(System.in));  
   StreamTokenizer in = new StreamTokenizer(br);  
   PrintWriter out = new PrintWriter(new OutputStreamWriter(System.out));  
   while (in.nextToken() != StreamTokenizer.TT_EOF) {  
      int n = (int) in.nval;  
      for (int i = 0; i < n; i++) {  
         in.nextToken();  
         double x = (double) in.nval;  
         in.nextToken();  
         double y = (double) in.nval;  
         points[i] = new Point(x, y);  
      }  
      Arrays.sort(points, 0, n, (a, b) -> a.x <= b.x ? -1 : 1);  
      double ans = nearest(0, n - 1);  
      out.println(String.format("%.4f", ans));  
      out.flush();  
   }  
}

```


例子:

```java
public class Main {
	public static void main(String[] args) {
		// System.in and System.out are input and output streams, respectively.
		InputReader in = new InputReader(System.in);

		// Read the number of test casese.		
		int T = in.nextInt();
		while (T-- > 0) {
			// Read the numbers a and b.
			int a = in.nextInt();
			int b = in.nextInt();
			
			// Compute the sum of a and b.
			int ans = a + b;
			System.out.println(ans);
		}
	}

	static class InputReader {
		public BufferedReader reader;
		public StringTokenizer tokenizer;

		public InputReader(InputStream stream) {
			reader = new BufferedReader(new InputStreamReader(stream), 32768);
			tokenizer = null;
		}

		public String next() {
			while (tokenizer == null || !tokenizer.hasMoreTokens()) {
				try {
				    tokenizer = new StringTokenizer(reader.readLine());
				} catch (IOException e) {
				    throw new RuntimeException(e);
				}
			}
			return tokenizer.nextToken();
		}

		public int nextInt() {
			return Integer.parseInt(next());
		}
	}
}

```

### FastIO

```java
import java.io.IOException;  
import java.io.InputStream;  
import java.util.Arrays;  
import java.util.InputMismatchException;  
  
public class FastIO {  
    InputStream is;  
  
    public FastIO(final InputStream is) {  
        this.is = is;  
    }  
  
    private byte[] inbuf = new byte[8192];  
    public int lenbuf = 0, ptrbuf = 0;  
  
    public int readByte() {  
        if (lenbuf == -1) {  
            throw new InputMismatchException();  
        }  
        if (ptrbuf >= lenbuf) {  
            ptrbuf = 0;  
            try {  
                lenbuf = is.read(inbuf);  
            } catch (IOException e) {  
                throw new InputMismatchException();  
            }  
            if (lenbuf <= 0) {  
                return -1;  
            }  
        }  
        return inbuf[ptrbuf++];  
    }  
  
    private boolean isSpaceChar(int c) {  
        return !(c >= 33 && c <= 126);  
    }  
  
    private int skip() {  
        int b;  
        while ((b = readByte()) != -1 && isSpaceChar(b)) ;  
        return b;  
    }  
  
    public double nextDouble() {  
        return Double.parseDouble(nextString());  
    }  
  
    public char nextChar() {  
        return (char) skip();  
    }  
  
    public String nextString() {  
        int b = skip();  
        StringBuilder sb = new StringBuilder();  
        while (!(isSpaceChar(b))) { // when nextLine, (isSpaceChar(b) && b != ' ')  
            sb.appendCodePoint(b);  
            b = readByte();  
        }  
        return sb.toString();  
    }  
  
    public char[] nextCharArray(int n) {  
        char[] buf = new char[n];  
        int b = skip(), p = 0;  
        while (p < n && !(isSpaceChar(b))) {  
            buf[p++] = (char) b;  
            b = readByte();  
        }  
        return n == p ? buf : Arrays.copyOf(buf, p);  
    }  
  
    public int[] nextArrayInt(int n) {  
        int[] a = new int[n];  
        for (int i = 0; i < n; i++) {  
            a[i] = nextInt();  
        }  
        return a;  
    }  
  
    public long[] nextArrayLong(int n) {  
        long[] a = new long[n];  
        for (int i = 0; i < n; i++) {  
            a[i] = nextLong();  
        }  
        return a;  
    }  
  
    public char[][] nextMapChar(int n, int m) {  
        char[][] map = new char[n][];  
        for (int i = 0; i < n; i++) {  
            map[i] = nextCharArray(m);  
        }  
        return map;  
    }  
  
    public int[][] nextMapInt(int n, int m) {  
        int[][] map = new int[n][];  
        for (int i = 0; i < n; i++) {  
            map[i] = nextArrayInt(m);  
        }  
        return map;  
    }  
  
    public int nextInt() {  
        int num = 0;  
        int b;  
        boolean minus = false;  
        while ((b = readByte()) != -1 && !((b >= '0' && b <= '9') || b == '-'))  
            ;  
        if (b == '-') {  
            minus = true;  
            b = readByte();  
        }  
  
        while (true) {  
            if (b >= '0' && b <= '9') {  
                num = num * 10 + (b - '0');  
            } else {  
                return minus ? -num : num;  
            }  
            b = readByte();  
        }  
    }  
  
    public long nextLong() {  
        long num = 0;  
        int b;  
        boolean minus = false;  
        while ((b = readByte()) != -1 && !((b >= '0' && b <= '9') || b == '-'))  
            ;  
        if (b == '-') {  
            minus = true;  
            b = readByte();  
        }  
  
        while (true) {  
            if (b >= '0' && b <= '9') {  
                num = num * 10 + (b - '0');  
            } else {  
                return minus ? -num : num;  
            }  
            b = readByte();  
        }  
    }  
}

```


more:
https://atcoder.jp/contests/arc143/submissions/32782389  
FastIO的代码 