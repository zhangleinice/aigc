import pytest  # 用于我们的单元测试


def format_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours > 0:
        return f"{hours}h{minutes}min{seconds}s"
    elif minutes > 0:
        return f"{minutes}min{seconds}s"
    else:
        return f"{seconds}s"


#下面，每个测试案例由传递给 @pytest.mark.parametrize 装饰器的元组表示，每个元组都包含输入和预期输出
@pytest.mark.parametrize("seconds, expected", [
    (0, "0s"),
    (1, "1s"),
    (60, "1min0s"),
    (3600, "1h0min0s"),
    (3601, "1h0min1s"),
    (3660, "1h1min0s"),
    (3661, "1h1min1s"),
    (7261, "2h1min1s"),
    (None, TypeError),
    ("string", TypeError)
])
def test_format_time(seconds, expected):
    # 我们使用 try/except 语句来捕获并处理函数中可能出现的异常
    try:
        result = format_time(seconds)
    except TypeError:
        result = TypeError

    # 我们使用 assert 语句来验证函数的行为是否与预期一致
    assert result == expected

format_time(66)