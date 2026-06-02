from citation_deep_research import rate_limit


class FakeTime:
    def __init__(self) -> None:
        self.now = 100.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def test_rate_limiter_waits_between_calls(monkeypatch) -> None:
    fake_time = FakeTime()
    monkeypatch.setattr(rate_limit, "time", fake_time)

    limiter = rate_limit.RateLimiter(requests_per_second=1.0)
    limiter.wait()
    limiter.wait()

    assert fake_time.slept == [1.0]
