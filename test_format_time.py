# 用Python写一个函数，进行时间格式化输出，比如：
# 输入  输出
# 1  1s
# 61  1min1s
# 要求仅需要格式化到小时(?h?min?s)，即可

def format_time(seconds):

    # 验证输入是否为整数类型
    if not isinstance(seconds, int):
        raise ValueError("输入必须为整数类型")

    # 处理负数情况
    if seconds < 0:
        return "-" + format_time(-seconds)

    if seconds == 0:
        return "0s"

    # 计算小时、分钟和秒数
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60

    # 格式化输出
    time_str = ""
    if hours > 0:
        time_str += f"{hours}h"
    if minutes > 0:
        time_str += f"{minutes}min"
    if remaining_seconds > 0:
        time_str += f"{remaining_seconds}s"

    return time_str

# 测试
print(format_time(1))   # 输出：1s
print(format_time(61))  # 输出：1min1s
print(format_time(3661)) # 输出：1h1min1s


# 测试用例

import pytest


# 增加负数和非整数校验
def test_format_time_seconds():
    assert format_time(0) == "0s"
    assert format_time(1) == "1s"
    assert format_time(59) == "59s"
    assert format_time(61) == "1min1s"
    assert format_time(119) == "1min59s"

def test_format_time_minutes():
    assert format_time(60) == "1min"
    assert format_time(119) == "1min59s"
    assert format_time(120) == "2min"
    assert format_time(179) == "2min59s"

def test_format_time_hours():
    assert format_time(3600) == "1h"
    assert format_time(3601) == "1h1s"
    assert format_time(3660) == "1h1min"
    assert format_time(3661) == "1h1min1s"
    assert format_time(7200) == "2h"
    assert format_time(7261) == "2h1min1s"

def test_format_time_negative():
    assert format_time(-1) == "-1s"
    assert format_time(-60) == "-1min"
    assert format_time(-3600) == "-1h"



