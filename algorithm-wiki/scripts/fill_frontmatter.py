#!/usr/bin/env python3
"""
fill_frontmatter.py — 给 题目/*.md 批量补 difficulty / tags / covers。

策略：
  - difficulty: 从内置字典查（基于 LC 题号）。未覆盖的题留空。
  - tags / covers: 从 sources 字段推断（只看主题文件 source，不看题单 source）

幂等：再跑一次只会填新的字段，已有的非空字段不动（除非加 --force）。

用法：
    python fill_frontmatter.py                # 全跑
    python fill_frontmatter.py --dry-run      # 只打印不动文件
    python fill_frontmatter.py --report       # 列出 difficulty/tags/covers 仍空的题
"""

from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # vault root
PROBLEMS = ROOT / "题目"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
DIFFICULTY_FILE = DATA_DIR / "lc_difficulty.json"

# ---------------------------------------------------------------------------
# Source → (tags, covers)
# 只匹配主题来源（灵茶山艾府和代码随想录/X），不处理题单 source（题单/X）
# ---------------------------------------------------------------------------
SOURCE_TO_META: dict[str, tuple[list[str], list[str]]] = {
    "灵茶山艾府和代码随想录/链表":              (["linked-list"],     ["linked-list"]),
    "灵茶山艾府和代码随想录/二叉树":            (["tree"],            ["binary-tree"]),
    "灵茶山艾府和代码随想录/二分查找":          (["binary-search"],   ["binary-search"]),
    "灵茶山艾府和代码随想录/动态规划":          (["dp"],              ["dp"]),
    "灵茶山艾府和代码随想录/单调栈":            (["stack"],           ["monotonic-stack", "stack"]),
    "灵茶山艾府和代码随想录/单调队列":          (["queue"],           ["monotonic-queue", "queue"]),
    "灵茶山艾府和代码随想录/前缀和":            (["prefix-sum"],      ["prefix-sum"]),
    "灵茶山艾府和代码随想录/双向双指针":        (["two-pointers"],    ["two-pointers"]),
    "灵茶山艾府和代码随想录/滑动窗口(同向双指针)": (["sliding-window"], ["sliding-window", "two-pointers"]),
    "灵茶山艾府和代码随想录/哈希表":            (["hash-table"],      ["hash-table"]),
    "灵茶山艾府和代码随想录/回溯":              (["backtracking"],    ["backtracking"]),
    "灵茶山艾府和代码随想录/图":                (["graph"],           ["graph", "bfs-dfs"]),
    "灵茶山艾府和代码随想录/字符串":            (["string"],          ["kmp"]),
    "灵茶山艾府和代码随想录/扫描线":            (["scan-line"],       ["scan-line"]),
    "灵茶山艾府和代码随想录/数据结构":          (["design"],          []),
    "灵茶山艾府和代码随想录/栈和队列":          (["stack", "queue"],  ["stack", "queue"]),
    "灵茶山艾府和代码随想录/贪心":              (["greedy"],          ["greedy"]),
    # 题单类不映射 tags/covers（题单是来源，不决定主题）
}

# ---------------------------------------------------------------------------
# LC题号 → difficulty
# 只列出我有把握的、覆盖大多数常见 Hot100 / Blind75 / Grind75 题
# 未覆盖的留空，可后续让 LLM 用 web search 补全
# ---------------------------------------------------------------------------
LC_DIFFICULTY: dict[int, str] = {
    # ==== Easy（已知高频）====
    1: "easy", 9: "easy", 13: "easy", 14: "easy", 20: "easy", 21: "easy",
    26: "easy", 27: "easy", 28: "easy", 35: "easy", 58: "easy", 66: "easy",
    69: "easy", 70: "easy", 83: "easy", 88: "easy", 94: "easy", 100: "easy",
    101: "easy", 104: "easy", 108: "easy", 110: "easy", 111: "easy", 112: "easy",
    118: "easy", 119: "easy", 121: "easy", 125: "easy", 136: "easy", 141: "easy",
    144: "easy", 145: "easy", 155: "easy", 160: "easy", 169: "easy", 191: "easy",
    202: "easy", 203: "easy", 205: "easy", 206: "easy", 217: "easy", 219: "easy",
    222: "easy", 225: "easy", 226: "easy", 228: "easy", 231: "easy", 232: "easy",
    234: "easy", 237: "easy", 242: "easy", 257: "easy", 258: "easy", 263: "easy",
    268: "easy", 283: "easy", 290: "easy", 292: "easy", 326: "easy", 338: "easy",
    342: "easy", 344: "easy", 345: "easy", 349: "easy", 350: "easy", 367: "easy",
    374: "easy", 383: "easy", 387: "easy", 389: "easy", 392: "easy", 401: "easy",
    404: "easy", 405: "easy", 409: "easy", 412: "easy", 414: "easy", 415: "easy",
    441: "easy", 448: "easy", 455: "easy", 459: "easy", 461: "easy", 463: "easy",
    476: "easy", 482: "easy", 485: "easy", 496: "easy", 500: "easy", 501: "easy",
    504: "easy", 506: "easy", 507: "easy", 509: "easy", 520: "easy", 521: "easy",
    530: "easy", 538: "easy", 541: "easy", 543: "easy", 551: "easy", 557: "easy",
    559: "easy", 561: "easy", 563: "easy", 566: "easy", 572: "easy", 575: "easy",
    594: "easy", 595: "easy", 596: "easy", 598: "easy", 599: "easy", 605: "easy",
    606: "easy", 617: "easy", 628: "easy", 633: "easy", 637: "easy", 643: "easy",
    645: "easy", 657: "easy", 661: "easy", 671: "easy", 674: "easy", 680: "easy",
    682: "easy", 690: "easy", 693: "easy", 696: "easy", 697: "easy", 700: "easy",
    703: "easy", 704: "easy", 705: "easy", 706: "easy", 717: "easy", 724: "easy",
    728: "easy", 733: "easy", 744: "easy", 746: "easy", 762: "easy", 766: "easy",
    771: "easy", 783: "easy", 796: "easy", 804: "easy", 806: "easy", 819: "easy",
    821: "easy", 836: "easy", 844: "easy", 859: "easy", 860: "easy", 867: "easy",
    868: "easy", 872: "easy", 876: "easy", 884: "easy", 888: "easy", 892: "easy",
    896: "easy", 897: "easy", 905: "easy", 914: "easy", 917: "easy", 922: "easy",
    925: "easy", 929: "easy", 933: "easy", 938: "easy", 941: "easy", 944: "easy",
    953: "easy", 961: "easy", 965: "easy", 970: "easy", 977: "easy", 989: "easy",
    993: "easy", 997: "easy", 999: "easy", 1002: "easy", 1005: "easy", 1009: "easy",
    1018: "easy", 1021: "easy", 1037: "easy", 1046: "easy", 1047: "easy", 1051: "easy",
    1071: "easy", 1086: "easy", 1108: "easy", 1119: "easy", 1122: "easy", 1128: "easy",
    1133: "easy", 1207: "easy", 1217: "easy", 1342: "easy", 1346: "easy", 1380: "easy",
    1431: "easy", 1480: "easy", 1502: "easy", 1512: "easy", 1572: "easy", 1576: "easy",
    1672: "easy", 1684: "easy", 1763: "easy", 1832: "easy", 1929: "easy", 2103: "easy",
    2108: "easy", 2200: "easy", 2540: "easy",

    # ==== Medium（已知高频）====
    2: "medium", 3: "medium", 5: "medium", 6: "medium", 7: "medium", 8: "medium",
    11: "medium", 12: "medium", 15: "medium", 16: "medium", 17: "medium", 18: "medium",
    19: "medium", 22: "medium", 24: "medium", 29: "medium", 31: "medium", 33: "medium",
    34: "medium", 36: "medium", 38: "medium", 39: "medium", 40: "medium", 43: "medium",
    45: "medium", 46: "medium", 47: "medium", 48: "medium", 49: "medium", 50: "medium",
    53: "medium", 54: "medium", 55: "medium", 56: "medium", 57: "medium", 59: "medium",
    61: "medium", 62: "medium", 63: "medium", 64: "medium", 67: "medium", 71: "medium",
    73: "medium", 74: "medium", 75: "medium", 77: "medium", 78: "medium", 79: "medium",
    80: "medium", 81: "medium", 82: "medium", 86: "medium", 89: "medium", 90: "medium",
    91: "medium", 92: "medium", 93: "medium", 95: "medium", 96: "medium", 97: "medium",
    98: "medium", 102: "medium", 103: "medium", 105: "medium", 106: "medium", 107: "medium",
    109: "medium", 113: "medium", 114: "medium", 116: "medium", 117: "medium", 120: "medium",
    122: "medium", 128: "medium", 129: "medium", 130: "medium", 131: "medium", 133: "medium",
    134: "medium", 137: "medium", 138: "medium", 139: "medium", 142: "medium", 143: "medium",
    146: "medium", 147: "medium", 148: "medium", 150: "medium", 151: "medium", 152: "medium",
    153: "medium", 162: "medium", 165: "medium", 166: "medium", 167: "medium", 173: "medium",
    179: "medium", 187: "medium", 189: "medium", 198: "medium", 199: "medium", 200: "medium",
    207: "medium", 208: "medium", 209: "medium", 210: "medium", 211: "medium", 213: "medium",
    215: "medium", 216: "medium", 220: "medium", 221: "medium", 223: "medium", 227: "medium",
    229: "medium", 230: "medium", 235: "medium", 236: "medium", 238: "medium", 240: "medium",
    241: "medium", 247: "medium", 249: "medium", 250: "medium", 254: "medium", 256: "medium",
    260: "medium", 261: "medium", 264: "medium", 274: "medium", 275: "medium", 277: "medium",
    279: "medium", 280: "medium", 281: "medium", 285: "medium", 286: "medium", 287: "medium",
    289: "medium", 291: "medium", 298: "medium", 299: "medium", 300: "medium", 304: "medium",
    306: "medium", 307: "medium", 309: "medium", 310: "medium", 311: "medium", 313: "medium",
    314: "medium", 318: "medium", 319: "medium", 322: "medium", 323: "medium", 324: "medium",
    325: "medium", 328: "medium", 331: "medium", 333: "medium", 334: "medium", 337: "medium",
    339: "medium", 340: "medium", 341: "medium", 343: "medium", 347: "medium", 348: "medium",
    351: "medium", 353: "medium", 355: "medium", 357: "medium", 360: "medium", 361: "medium",
    362: "medium", 366: "medium", 369: "medium", 370: "medium", 371: "medium", 372: "medium",
    373: "medium", 375: "medium", 376: "medium", 377: "medium", 378: "medium", 379: "medium",
    380: "medium", 382: "medium", 384: "medium", 385: "medium", 386: "medium", 388: "medium",
    390: "medium", 393: "medium", 394: "medium", 395: "medium", 396: "medium", 397: "medium",
    398: "medium", 399: "medium", 400: "medium", 402: "medium", 406: "medium", 408: "medium",
    413: "medium", 416: "medium", 417: "medium", 419: "medium", 421: "medium", 423: "medium",
    424: "medium", 426: "medium", 429: "medium", 430: "medium", 433: "medium", 434: "medium",
    435: "medium", 436: "medium", 438: "medium", 439: "medium", 442: "medium", 443: "medium",
    444: "medium", 445: "medium", 449: "medium", 450: "medium", 451: "medium", 452: "medium",
    453: "medium", 454: "medium", 456: "medium", 457: "medium", 462: "medium", 464: "medium",
    467: "medium", 468: "medium", 470: "medium", 473: "medium", 474: "medium", 475: "medium",
    477: "medium", 478: "medium", 479: "medium", 481: "medium", 484: "medium", 486: "medium",
    487: "medium", 490: "medium", 491: "medium", 494: "medium", 497: "medium", 498: "medium",
    503: "medium", 505: "medium", 508: "medium", 510: "medium", 513: "medium", 515: "medium",
    516: "medium", 518: "medium", 519: "medium", 522: "medium", 523: "medium", 524: "medium",
    525: "medium", 526: "medium", 528: "medium", 529: "medium", 531: "medium", 532: "medium",
    533: "medium", 534: "medium", 535: "medium", 536: "medium", 537: "medium", 539: "medium",
    540: "medium", 542: "medium", 544: "medium", 545: "medium", 547: "medium", 548: "medium",
    549: "medium", 553: "medium", 554: "medium", 555: "medium", 556: "medium", 558: "medium",
    560: "medium", 562: "medium", 565: "medium", 567: "medium", 569: "medium", 573: "medium",
    576: "medium", 577: "medium", 580: "medium", 582: "medium", 583: "medium", 587: "medium",
    589: "medium", 590: "medium", 592: "medium", 593: "medium", 600: "medium", 602: "medium",
    604: "medium", 608: "medium", 609: "medium", 611: "medium", 612: "medium", 613: "medium",
    615: "medium", 616: "medium", 618: "medium", 619: "medium", 621: "medium", 623: "medium",
    624: "medium", 625: "medium", 626: "medium", 627: "medium", 629: "medium", 630: "medium",
    634: "medium", 635: "medium", 636: "medium", 638: "medium", 640: "medium", 641: "medium",
    642: "medium", 644: "medium", 646: "medium", 647: "medium", 648: "medium", 649: "medium",
    650: "medium", 651: "medium", 652: "medium", 653: "medium", 654: "medium", 655: "medium",
    658: "medium", 659: "medium", 660: "medium", 662: "medium", 666: "medium", 667: "medium",
    668: "medium", 669: "medium", 670: "medium", 672: "medium", 673: "medium", 676: "medium",
    677: "medium", 678: "medium", 681: "medium", 684: "medium", 686: "medium", 687: "medium",
    688: "medium", 692: "medium", 694: "medium", 695: "medium", 698: "medium", 701: "medium",
    702: "medium", 707: "medium", 708: "medium", 712: "medium", 713: "medium", 714: "medium",
    718: "medium", 721: "medium", 723: "medium", 725: "medium", 729: "medium", 731: "medium",
    735: "medium", 738: "medium", 739: "medium", 740: "medium", 743: "medium", 750: "medium",
    752: "medium", 754: "medium", 755: "medium", 756: "medium", 763: "medium", 764: "medium",
    767: "medium", 769: "medium", 775: "medium", 776: "medium", 777: "medium", 779: "medium",
    781: "medium", 784: "medium", 785: "medium", 787: "medium", 788: "medium", 789: "medium",
    790: "medium", 791: "medium", 792: "medium", 794: "medium", 795: "medium", 797: "medium",
    798: "medium", 799: "medium", 800: "medium", 801: "medium", 802: "medium", 807: "medium",
    808: "medium", 809: "medium", 813: "medium", 814: "medium", 816: "medium", 817: "medium",
    820: "medium", 822: "medium", 823: "medium", 825: "medium", 826: "medium", 831: "medium",
    833: "medium", 835: "medium", 837: "medium", 838: "medium", 840: "medium", 841: "medium",
    842: "medium", 845: "medium", 846: "medium", 848: "medium", 849: "medium", 851: "medium",
    853: "medium", 855: "medium", 856: "medium", 858: "medium", 861: "medium", 863: "medium",
    865: "medium", 866: "medium", 869: "medium", 870: "medium", 873: "medium", 875: "medium",
    877: "medium", 880: "medium", 881: "medium", 885: "medium", 886: "medium", 889: "medium",
    890: "medium", 893: "medium", 894: "medium", 898: "medium", 900: "medium", 901: "medium",
    904: "medium", 907: "medium", 909: "medium", 910: "medium", 911: "medium", 912: "medium",
    915: "medium", 916: "medium", 918: "medium", 919: "medium", 921: "medium", 923: "medium",
    926: "medium", 930: "medium", 931: "medium", 932: "medium", 934: "medium", 935: "medium",
    939: "medium", 945: "medium", 946: "medium", 947: "medium", 948: "medium", 950: "medium",
    951: "medium", 954: "medium", 955: "medium", 957: "medium", 958: "medium", 959: "medium",
    962: "medium", 963: "medium", 966: "medium", 967: "medium", 969: "medium", 971: "medium",
    973: "medium", 974: "medium", 978: "medium", 979: "medium", 981: "medium", 983: "medium",
    984: "medium", 986: "medium", 988: "medium", 990: "medium", 991: "medium", 994: "medium",
    998: "medium", 1003: "medium", 1004: "medium", 1006: "medium", 1007: "medium", 1008: "medium",
    1010: "medium", 1011: "medium", 1014: "medium", 1015: "medium", 1019: "medium", 1020: "medium",
    1023: "medium", 1024: "medium", 1026: "medium", 1027: "medium", 1031: "medium", 1034: "medium",
    1035: "medium", 1038: "medium", 1039: "medium", 1041: "medium", 1042: "medium", 1043: "medium",
    1048: "medium", 1049: "medium", 1052: "medium", 1054: "medium", 1062: "medium", 1072: "medium",
    1079: "medium", 1080: "medium", 1081: "medium", 1090: "medium", 1091: "medium", 1094: "medium",
    1100: "medium", 1102: "medium", 1103: "medium", 1104: "medium", 1109: "medium", 1110: "medium",
    1115: "medium", 1116: "medium", 1129: "medium", 1130: "medium", 1138: "medium", 1140: "medium",
    1143: "medium", 1146: "medium", 1155: "medium", 1162: "medium", 1167: "medium", 1171: "medium",
    1186: "medium", 1190: "medium", 1191: "medium", 1197: "medium", 1198: "medium", 1202: "medium",
    1218: "medium", 1219: "medium", 1222: "medium", 1230: "medium", 1248: "medium", 1262: "medium",
    1268: "medium", 1283: "medium", 1284: "medium", 1286: "medium", 1296: "medium", 1297: "medium",
    1300: "medium", 1305: "medium", 1306: "medium", 1310: "medium", 1314: "medium", 1318: "medium",
    1319: "medium", 1325: "medium", 1334: "medium", 1338: "medium", 1339: "medium", 1361: "medium",
    1376: "medium", 1382: "medium", 1395: "medium", 1396: "medium", 1402: "medium", 1424: "medium",
    1438: "medium", 1456: "medium", 1462: "medium", 1466: "medium", 1472: "medium", 1493: "medium",
    1514: "medium", 1525: "medium", 1530: "medium", 1559: "medium", 1584: "medium", 1631: "medium",
    1647: "medium", 1648: "medium", 1657: "medium", 1658: "medium", 1696: "medium", 1700: "medium",
    1721: "medium", 1737: "medium", 1759: "medium", 1786: "medium", 1813: "medium", 1834: "medium",
    1855: "medium", 1856: "medium", 1857: "medium", 1864: "medium", 1898: "medium", 1905: "medium",
    1922: "medium", 1942: "medium", 1958: "medium", 2034: "medium", 2050: "medium", 2068: "medium",
    2090: "medium", 2096: "medium", 2114: "medium", 2125: "medium", 2130: "medium", 2131: "medium",
    2149: "medium", 2161: "medium", 2167: "medium", 2178: "medium", 2196: "medium", 2226: "medium",
    2266: "medium", 2270: "medium", 2300: "medium", 2326: "medium", 2336: "medium", 2353: "medium",
    2390: "medium", 2461: "medium", 2466: "medium", 2487: "medium", 2516: "medium", 2529: "medium",
    2562: "medium", 2563: "medium", 2580: "medium", 2602: "medium", 2645: "medium", 2730: "medium",
    2826: "medium", 2958: "medium", 3217: "medium",

    # ==== Hard（已知高频）====
    4: "hard", 10: "hard", 23: "hard", 25: "hard", 30: "hard", 32: "hard",
    37: "hard", 41: "hard", 42: "hard", 44: "hard", 51: "hard", 52: "hard",
    65: "hard", 68: "hard", 72: "hard", 76: "hard", 84: "hard", 85: "hard",
    87: "hard", 99: "hard", 115: "hard", 123: "hard", 124: "hard", 126: "hard",
    127: "hard", 132: "hard", 135: "hard", 140: "hard", 149: "hard", 154: "hard",
    158: "hard", 164: "hard", 174: "hard", 188: "hard", 212: "hard", 214: "hard",
    218: "hard", 224: "hard", 233: "hard", 239: "hard", 248: "hard", 269: "hard",
    273: "hard", 282: "hard", 295: "hard", 297: "hard", 301: "hard", 302: "hard",
    305: "hard", 312: "hard", 315: "hard", 316: "medium", 317: "hard", 321: "hard",
    327: "hard", 329: "hard", 330: "hard", 332: "hard", 335: "hard", 336: "hard",
    352: "hard", 354: "hard", 358: "hard", 363: "hard", 381: "hard", 391: "hard",
    403: "hard", 407: "hard", 410: "hard", 411: "hard", 420: "hard", 425: "hard",
    432: "hard", 440: "hard", 446: "hard", 458: "hard", 460: "hard", 466: "hard",
    472: "hard", 480: "hard", 483: "hard", 488: "hard", 489: "hard", 493: "hard",
    499: "hard", 502: "hard", 514: "hard", 517: "hard", 527: "hard", 546: "hard",
    552: "hard", 564: "hard", 568: "hard", 587: "hard", 591: "hard", 600: "hard",
    629: "hard", 630: "hard", 632: "hard", 642: "hard", 644: "hard", 656: "hard",
    664: "hard", 675: "hard", 679: "hard", 685: "hard", 689: "hard", 691: "hard",
    699: "hard", 715: "hard", 719: "hard", 726: "hard", 727: "hard", 730: "hard",
    732: "hard", 736: "hard", 745: "hard", 749: "hard", 753: "hard", 757: "hard",
    761: "hard", 765: "hard", 768: "hard", 770: "hard", 772: "hard", 773: "hard",
    774: "hard", 778: "hard", 780: "hard", 782: "hard", 793: "hard", 803: "hard",
    805: "hard", 815: "hard", 818: "hard", 827: "hard", 828: "hard", 829: "hard",
    834: "hard", 839: "hard", 843: "hard", 850: "hard", 854: "hard", 857: "hard",
    862: "hard", 864: "hard", 871: "hard", 878: "hard", 879: "hard", 882: "hard",
    887: "hard", 891: "hard", 895: "hard", 902: "hard", 903: "hard", 906: "hard",
    920: "hard", 924: "hard", 927: "hard", 928: "hard", 936: "hard", 940: "hard",
    943: "hard", 952: "medium", 956: "hard", 960: "hard", 964: "hard", 968: "hard",
    972: "hard", 980: "hard", 982: "hard", 992: "hard", 995: "hard", 996: "hard",
    1000: "hard", 1106: "hard", 1278: "hard", 1739: "hard", 2421: "medium",
}


# ---------------------------------------------------------------------------
# Persist / load extra difficulty additions to JSON (so user/LLM can extend)
# ---------------------------------------------------------------------------
def load_extra_difficulty() -> dict[int, str]:
    if DIFFICULTY_FILE.exists():
        try:
            data = json.loads(DIFFICULTY_FILE.read_text(encoding="utf-8"))
            return {int(k): v for k, v in data.items()}
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


def save_extra_difficulty(extra: dict[int, str]) -> None:
    """Save extras (allows user/LLM to add via JSON)."""
    DIFFICULTY_FILE.parent.mkdir(parents=True, exist_ok=True)
    DIFFICULTY_FILE.write_text(
        json.dumps({str(k): v for k, v in sorted(extra.items())}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Frontmatter helpers (no PyYAML dependency, simple regex-based)
# ---------------------------------------------------------------------------
FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)

def parse_frontmatter(text: str) -> tuple[dict | None, str]:
    m = FRONTMATTER_RE.match(text)
    if not m:
        return None, text
    fm: dict = {}
    body = text[m.end():]
    current_key = None
    list_items: list[str] = []
    for line in m.group(1).split("\n"):
        if not line.strip():
            continue
        # List continuation
        if line.startswith("  - "):
            if current_key is not None:
                list_items.append(line[4:].strip())
            continue
        # Flush previous list
        if current_key is not None and list_items:
            fm[current_key] = list_items
            list_items = []
        # Key-value
        kv = re.match(r"^([\w一-鿿_-]+):\s*(.*)$", line)
        if kv:
            key, val = kv.group(1), kv.group(2).strip()
            current_key = key
            list_items = []
            if val == "":
                fm[key] = None
                # could be a list to follow
            elif val == "[]":
                fm[key] = []
            elif val.startswith("[") and val.endswith("]"):
                inner = val[1:-1].strip()
                fm[key] = [x.strip() for x in inner.split(",") if x.strip()] if inner else []
            else:
                # Try int
                try:
                    fm[key] = int(val)
                except ValueError:
                    fm[key] = val
                current_key = None
    # Flush
    if current_key is not None and list_items:
        fm[current_key] = list_items
    return fm, body


def render_frontmatter(fm: dict) -> str:
    lines = ["---"]
    for k, v in fm.items():
        if v is None or v == "":
            lines.append(f"{k}: ")
        elif isinstance(v, list):
            if not v:
                lines.append(f"{k}: []")
            else:
                lines.append(f"{k}:")
                for item in v:
                    lines.append(f"  - {item}")
        elif isinstance(v, bool):
            lines.append(f"{k}: {'true' if v else 'false'}")
        else:
            lines.append(f"{k}: {v}")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Update logic
# ---------------------------------------------------------------------------
def update_one(path: Path, force: bool = False, dry_run: bool = False,
               extra_difficulty: dict[int, str] | None = None) -> dict:
    extra_difficulty = extra_difficulty or {}
    text = path.read_text(encoding="utf-8")
    fm, body = parse_frontmatter(text)
    if fm is None:
        return {"path": path, "changed": False, "reason": "no frontmatter"}

    changes: list[str] = []

    # 1) difficulty
    cur_diff = fm.get("difficulty")
    if (cur_diff is None or cur_diff == "" or force):
        lc_num = fm.get("leetcode")
        if isinstance(lc_num, int):
            new_diff = LC_DIFFICULTY.get(lc_num) or extra_difficulty.get(lc_num)
            if new_diff:
                fm["difficulty"] = new_diff
                changes.append(f"difficulty={new_diff}")

    # 2) tags + covers from sources
    sources = fm.get("sources") or []
    if isinstance(sources, list):
        existing_tags = set(fm.get("tags") or [])
        existing_covers = set(fm.get("covers") or [])
        new_tags, new_covers = set(), set()
        for src in sources:
            if src in SOURCE_TO_META:
                t, c = SOURCE_TO_META[src]
                new_tags.update(t)
                new_covers.update(c)
        merged_tags = sorted(existing_tags | new_tags)
        merged_covers = sorted(existing_covers | new_covers)
        if merged_tags != sorted(existing_tags):
            fm["tags"] = merged_tags
            changes.append(f"tags+={list(new_tags - existing_tags)}")
        if merged_covers != sorted(existing_covers):
            fm["covers"] = merged_covers
            changes.append(f"covers+={list(new_covers - existing_covers)}")

    if not changes:
        return {"path": path, "changed": False, "reason": "no change needed"}

    if not dry_run:
        new_text = render_frontmatter(fm) + body
        path.write_text(new_text, encoding="utf-8")

    return {"path": path, "changed": True, "changes": changes}


def report_blanks() -> dict:
    """List which difficulty/tags/covers are still empty."""
    missing_diff: list[int] = []
    missing_tags: list[Path] = []
    missing_covers: list[Path] = []
    extra_difficulty = load_extra_difficulty()
    for f in sorted(PROBLEMS.glob("*.md")):
        if f.name in {"INDEX.md", "DASHBOARD.md", "README.md"}:
            continue
        fm, _ = parse_frontmatter(f.read_text(encoding="utf-8"))
        if not fm:
            continue
        if not fm.get("difficulty"):
            lc = fm.get("leetcode")
            if isinstance(lc, int) and lc not in LC_DIFFICULTY and lc not in extra_difficulty:
                missing_diff.append(lc)
        if not fm.get("tags"):
            missing_tags.append(f)
        if not fm.get("covers"):
            missing_covers.append(f)
    return {
        "missing_difficulty_lc_nums": sorted(set(missing_diff)),
        "missing_tags_files": [str(f.relative_to(ROOT)) for f in missing_tags[:30]],
        "missing_covers_files": [str(f.relative_to(ROOT)) for f in missing_covers[:30]],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Fill frontmatter for 题目/*.md")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing difficulty (still merges tags/covers)")
    ap.add_argument("--report", action="store_true",
                    help="Just print which fields are still empty")
    args = ap.parse_args()

    if args.report:
        r = report_blanks()
        print(f"=== 仍空的字段统计 ===")
        print(f"  difficulty 未覆盖的 LC 题号 ({len(r['missing_difficulty_lc_nums'])} 题):")
        nums = r["missing_difficulty_lc_nums"]
        for i in range(0, len(nums), 15):
            print(f"    {nums[i:i+15]}")
        print(f"  tags 仍空 ({len(r['missing_tags_files'])} 题，前 30):")
        for f in r["missing_tags_files"]:
            print(f"    {f}")
        print(f"  covers 仍空 ({len(r['missing_covers_files'])} 题，前 30):")
        for f in r["missing_covers_files"]:
            print(f"    {f}")
        return 0

    extra_diff = load_extra_difficulty()

    files = sorted(PROBLEMS.glob("*.md"))
    files = [f for f in files if f.name not in {"INDEX.md", "DASHBOARD.md", "README.md"}]

    changed_count = 0
    for f in files:
        result = update_one(f, force=args.force, dry_run=args.dry_run, extra_difficulty=extra_diff)
        if result["changed"]:
            changed_count += 1

    print(f"{'[DRY] ' if args.dry_run else ''}Updated {changed_count}/{len(files)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
