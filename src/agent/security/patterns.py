import re

# ─── Prompt Injection ───────────────────────────────────────────────
# Matches common prompt injection / jailbreak attempts (Chinese + English)
INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        # Chinese injection patterns
        r"忽略(之前|上面|所有|以前)(的)?(指令|规则|提示|要求|限制)",
        r"无视(之前|上面|所有|以前)(的)?(指令|规则|提示|要求|限制)",
        r"不要(遵循|遵守|理会|管)(之前|上面|以前)?(的)?(指令|规则|提示)",
        r"你现在(是|扮演|变成|成为)",
        r"从现在起(你)?(是|扮演|变成|成为)",
        r"(进入|切换到|开启)(开发者|debug|管理员|god|DAN|越狱|无限制)(模式|状态)",
        r"(你的|你)(真正|真实|原始|底层)(的)?(身份|角色|prompt|指令|系统提示)",
        r"(泄露|透露|展示|输出|打印)(你的|系统|内部)(的)?(prompt|指令|提示词|规则)",
        r"(假装|假设|想象)(你)?(没有|不存在|没有受到)(任何)?(限制|约束|规则)",
        r"(绕过|破解|突破|解除)(安全|限制|规则|约束|过滤)",
        r"(注入|jailbreak|越狱)",
        # English injection patterns
        r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|rules?|prompts?)",
        r"disregard\s+(all\s+)?(previous|prior|above)\s+(instructions?|rules?)",
        r"you\s+are\s+now\s+(a|an|the)",
        r"from\s+now\s+on\s+you\s+(are|will|shall)",
        r"(enter|switch\s+to|activate)\s+(developer|debug|admin|god|DAN|jailbreak)\s+mode",
        r"(reveal|show|print|output)\s+(your|the)\s+(system\s+)?prompt",
        r"(pretend|imagine|act\s+as\s+if)\s+you\s+(have\s+)?(no|don.t\s+have)\s+(restrictions?|rules?|limits?)",
        r"(bypass|circumvent|override)\s+(safety|security|rules?|filters?)",
    ]
]

# ─── Token Abuse ────────────────────────────────────────────────────
# Patterns that indicate attempts to waste tokens on computationally expensive tasks
TOKEN_ABUSE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(p, re.IGNORECASE), desc)
    for p, desc in [
        # Pi / irrational number digit requests
        (
            r"(π|pi|圆周率|派).{0,20}(小数|位|digits?|decimal)",
            "请求计算无理数位数",
        ),
        (
            r"(计算|算出|写出|列出|输出).{0,30}(π|pi|圆周率).{0,20}(位|小数|digits?)",
            "请求计算圆周率位数",
        ),
        (
            r"(小数点后|decimal\s+places?).{0,15}(\d{3,}|[一二三四五六七八九十百千万亿]{2,})",
            "请求超长小数位",
        ),
        # Math conjecture / proof requests
        (
            r"(证明|求证|论证|prove|demonstrate).{0,15}(哥德巴赫|费马|黎曼|四色|NP.{0,5}P|Goldbach|Fermat|Riemann|collatz|角谷|冰雹)",
            "请求证明数学猜想",
        ),
        (
            r"(证明|求证|论证).{0,10}(定理|猜想|假设|命题).{0,20}(详细|完整|严谨|严格)",
            "请求详细数学证明",
        ),
        # Listing large sequences
        (
            r"(列出|输出|写出|打印|生成|list|print).{0,20}(前|所有|全部|前\d*个).{0,10}(\d{3,}|[一二三四五六七八九十百千万]{2,}).{0,10}(素数|质数|斐波那契|fibonacci|因数|约数|因子)",
            "请求列出大量数学序列",
        ),
        (
            r"(前|前\d*个|\d{4,}个).{0,8}(素数|质数|斐波那契|fibonacci|因数|约数)",
            "请求大量数学序列",
        ),
        # Extreme computation requests
        (
            r"(计算|算|求).{0,20}(\d{10,}|[一二三四五六七八九十百千万亿]{5,}).{0,10}(阶乘|次方|幂|乘积|和|累加|连乘)",
            "请求大规模计算",
        ),
        (
            r"(factorial|!|power|exponent).{0,15}(\d{5,})",
            "请求大规模计算",
        ),
        # Long code generation
        (
            r"(写出|生成|编写|写一个|implement).{0,20}(\d{3,}|[一二三四五六七八九十百千万]{2,}).{0,10}(行|行代码|lines?|完整|全部)",
            "请求生成大量代码",
        ),
    ]
]

# ─── Off-Topic Keywords ─────────────────────────────────────────────
# Keywords that indicate non-learning-related topics
OFF_TOPIC_KEYWORD_GROUPS: dict[str, list[str]] = {
    "娱乐八卦": [
        "明星八卦", "绯闻", "追星", "饭圈", "偶像剧", "综艺节目",
        "选秀", "出道", "塌房", "饭拍", "应援", "打榜",
    ],
    "游戏攻略": [
        "游戏攻略", "通关秘籍", "装备搭配", "加点方案", "副本攻略",
        "PVP", "PVE", "开黑", "上分", "段位", "排位赛",
        "手游", "网游", "端游", "氪金", "抽卡", "十连抽",
    ],
    "政治时事": [
        "政治立场", "党派", "选举", "投票", "政权", "政变",
        "战争", "军事行动", "制裁", "贸易战",
    ],
    "算命占卜": [
        "算命", "占卜", "星座运势", "塔罗牌", "风水", "八字",
        "紫微斗数", "面相", "手相", "算卦", "抽签",
    ],
    "成人内容": [
        "色情", "成人小说", "美女", "裸体", "性行为", "性教育", "成人用品",
    ],
}

# ─── Image Abuse ────────────────────────────────────────────────────
# Patterns indicating attempts to access non-knowledge-base images
IMAGE_ABUSE_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"(显示|打开|查看|访问|给我看).{0,20}(http|https|ftp)://",
        r"(下载|抓取|爬取|获取).{0,15}(图片|照片|image|photo|pic)",
        r"(搜索|查找|找).{0,15}(图片|照片|image|photo|pic|壁纸|表情包|meme)",
        r"(生成|画|制作|create|generate|draw).{0,10}(图片|图像|画作|插画)",
        r"(看|查看|打开|显示).{0,10}(/etc/|/root/|/home/|/var/|/tmp/|C:\\|D:\\)",
        r"(看|查看|打开|显示).{0,10}\.\./",
    ]
]
