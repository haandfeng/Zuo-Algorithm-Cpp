```java
import java.util.*;

public class Solution {

    public static List<String> finalRiskScores(
            List<String> transaction_list,
            List<String> rules_list,
            List<String> merchants_list
    ) {
        // 1) init scores in insertion order (so output is stable)
        Map<String, Long> score = new LinkedHashMap<>();
        for (String s : merchants_list) {
            String[] p = s.split(",");
            String mid = p[0].trim();
            long init = Long.parseLong(p[1].trim());
            score.put(mid, init);
        }

        // 2) state: counts
        Map<String, Integer> cntPair = new HashMap<>();
        Map<String, Integer> cntPairHour = new HashMap<>();

        // 3) process transactions in order
        for (int i = 0; i < transaction_list.size(); i++) {
            String[] t = transaction_list.get(i).split(",");
            String merchant = t[0].trim();
            int amount = Integer.parseInt(t[1].trim());
            String customer = t[2].trim();
            int hour = Integer.parseInt(t[3].trim());

            String[] r = rules_list.get(i).split(",");
            int minAmount = Integer.parseInt(r[0].trim());
            int mul = Integer.parseInt(r[1].trim());
            int add = Integer.parseInt(r[2].trim());
            int penalty = Integer.parseInt(r[3].trim());

            // build keys
            String pairKey = merchant + "|" + customer;
            String pairHourKey = merchant + "|" + customer + "|" + hour;

            // update counts (then use updated value)
            int pairCount = cntPair.getOrDefault(pairKey, 0) + 1;
            cntPair.put(pairKey, pairCount);

            int pairHourCount = cntPairHour.getOrDefault(pairHourKey, 0) + 1;
            cntPairHour.put(pairHourKey, pairHourCount);

            // ensure score exists (if needed)
            score.putIfAbsent(merchant, 0L);
            long cur = score.get(merchant);

            // Rule 1
            if (amount > minAmount) cur = cur * mul;

            // Rule 2
            if (pairCount >= 4) cur = cur + add;

            // Rule 3
            if (pairHourCount >= 3) {
                if (hour >= 12 && hour <= 17) cur = cur + penalty;
                else if ((hour >= 9 && hour <= 11) || (hour >= 18 && hour <= 22)) cur = cur - penalty;
            }

            score.put(merchant, cur);
        }

        // 4) output
        List<String> out = new ArrayList<>();
        for (Map.Entry<String, Long> e : score.entrySet()) {
            out.add(e.getKey() + "," + e.getValue());
        }
        return out;
    }
}
```



问题描述
您需要设计一个风险评分系统，根据一系列交易数据和动态规则来计算商户的最终风险评分。系统处理三份输入数据：交易列表、规则列表和商户初始评分列表。您需要按顺序处理每笔交易，并根据以下规则更新相应商户的风险评分。

输入数据
List<String> transaction_list

描述：一个包含交易记录的字符串列表。

格式：每条字符串由逗号分隔，包含四个部分："商户ID,交易金额,客户ID,交易小时"。

数据类型：String, int, String, int。

示例：

{
"merchant_1,1400,customer_1,10",
"merchant_1,1000,customer_1,10",
"merchant_2,2000,customer_1,15",
"merchant_2,500,customer_2,22"
}
List<String> rules_list

描述：一个与 transaction_list 一一对应的规则列表。rules_list 中的第 i 条规则适用于 transaction_list 中的第 i 笔交易。

格式：每条字符串由逗号分隔，包含四个部分："最低交易金额,乘法因子,加法数值,罚金"。

数据类型：int, int, int, int。

示例：

{
"1000,2,9,15", // 对应第一条交易
"2000,3,5,9", // 对应第二条交易
"500,3,8,19", // 对应第三条交易
"1600,2,17,9" // 对应第四条交易
}
List<String> merchants_list

描述：一个包含商户及其初始风险评分的列表。

格式：每条字符串由逗号分隔，包含两个部分："商户ID,初始风险评分"。

数据类型：String, int。

示例：

{
"merchant_1,20",
"merchant_2,10"
}
计算规则
你需要按顺序遍历 transaction_list 中的每一笔交易，并根据以下三条规则，依次更新该交易对应商户的风险评分：

高额交易乘数规则：

如果当前交易的 交易金额 大于其对应规则中的 最低交易金额，则将该商户当前的风险评分乘以对应规则的 乘法因子。

高频客户累加规则：

对于同一个客户和同一个商户的组合，当交易次数超过3次时（即从第4笔交易开始），在处理这笔以及后续每一笔交易时，都将该商户的风险评分加上该笔交易对应规则的 加法数值。

小时内高频交易规则：

对于同一个客户和同一个商户在同一个小时内的交易，当交易次数达到第3次或更多时（即从第3笔交易开始），根据交易时间进行如下操作：

如果交易时间在 12:00-17:00 之间（包含边界），则将商户的风险评分加上该笔交易对应规则的 罚金。

如果交易时间在 9:00-11:00 或 18:00-22:00 之间（包含边界），则将商户的风险评分减去该笔交易对应规则的 罚金。

在其他时间段，风险评分不变。

输出格式
一个 List<String>，其中每个字符串包含商户ID和其最终计算出的风险评分。

格式："商户ID,最终风险评分"。

</hide>
求加米求加米，急需看面经！！