"""设置对象的行为测试（工单 #2）。

seam 是 ``Settings`` 的公开构造：所有可调项都有默认值，能从环境变量覆盖，
密钥留空时对象仍能构造成功（构造过程不联网、不读盘上的密钥）。
"""

import pytest
from pydantic import ValidationError

from rzyl_core.settings import Settings


def _load_settings() -> Settings:
    """构造设置对象，并显式不读 .env——测试只认 monkeypatch 进来的环境变量。

    ``_env_file`` 是 pydantic-settings 的运行时开关，类型签名里没有，故忽略告警。
    """
    return Settings(_env_file=None)  # pyright: ignore[reportCallIssue]


def test_settings_have_documented_defaults() -> None:
    settings = _load_settings()

    assert settings.window_message_limit == 30
    assert settings.window_minutes == 5
    assert settings.retention_days == 30
    assert settings.push_hour == 22
    assert settings.push_minute == 0
    assert settings.confidence_threshold == 0.7
    assert settings.group_whitelist == []
    assert settings.timezone == "Asia/Shanghai"


def test_settings_build_without_any_api_key() -> None:
    """没有密钥也要能构造：第一阶段全程不需要 API key。"""
    settings = _load_settings()

    assert settings.chat_api_key == ""
    assert settings.embedding_api_key == ""
    assert settings.chat_base_url
    assert settings.chat_model
    assert settings.embedding_base_url
    assert settings.embedding_model


def test_settings_read_overridable_values_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("RZYL_WINDOW_MESSAGE_LIMIT", "50")
    monkeypatch.setenv("RZYL_WINDOW_MINUTES", "10")
    monkeypatch.setenv("RZYL_RETENTION_DAYS", "7")
    monkeypatch.setenv("RZYL_PUSH_HOUR", "21")
    monkeypatch.setenv("RZYL_CONFIDENCE_THRESHOLD", "0.9")
    monkeypatch.setenv("RZYL_GROUP_WHITELIST", "123456,789012")
    monkeypatch.setenv("RZYL_CHAT_API_KEY", "chat-secret")
    monkeypatch.setenv("RZYL_EMBEDDING_API_KEY", "embed-secret")
    monkeypatch.setenv("RZYL_DATABASE_URL", "sqlite+aiosqlite:///tmp/other.db")

    settings = _load_settings()

    assert settings.window_message_limit == 50
    assert settings.window_minutes == 10
    assert settings.retention_days == 7
    assert settings.push_hour == 21
    assert settings.confidence_threshold == 0.9
    assert settings.group_whitelist == [123456, 789012]
    assert settings.chat_api_key == "chat-secret"
    assert settings.embedding_api_key == "embed-secret"
    assert settings.database_url == "sqlite+aiosqlite:///tmp/other.db"


def test_group_whitelist_accepts_json_list(monkeypatch) -> None:
    """复杂类型也接受 JSON 写法，和 pydantic-settings 的默认行为一致。"""
    monkeypatch.setenv("RZYL_GROUP_WHITELIST", "[111, 222]")

    assert _load_settings().group_whitelist == [111, 222]


def test_pricing_is_configurable_for_cost_estimation(monkeypatch) -> None:
    monkeypatch.setenv("RZYL_CHAT_INPUT_PRICE", "1.5")
    monkeypatch.setenv("RZYL_CHAT_OUTPUT_PRICE", "3.0")

    settings = _load_settings()

    assert settings.chat_input_price == 1.5
    assert settings.chat_output_price == 3.0


# —— 里程碑 2：实时链路的可调项 ——


def test_realtime_loop_and_backfill_defaults_are_configured() -> None:
    settings = _load_settings()

    # 三个后台循环的间隔都可配，都有合理默认。
    assert settings.window_flush_seconds > 0
    assert settings.dead_letter_retry_seconds > 0
    assert settings.dead_letter_max_retries > 0
    # 掉线回补的两条护栏：至多 24 小时或至多 N 条。
    assert settings.backfill_max_hours == 24
    assert settings.backfill_max_messages > 0


def test_chat_timeout_default_is_generous_enough_for_slow_models() -> None:
    # 实测 Qwen3.5-4B 单次可到 50 秒以上，默认 30 秒会直接把请求打死。
    assert _load_settings().chat_timeout >= 60


def test_chat_extra_body_defaults_to_empty() -> None:
    assert _load_settings().chat_extra_body == {}


def test_chat_extra_body_parses_a_json_object(monkeypatch) -> None:
    monkeypatch.setenv("RZYL_CHAT_EXTRA_BODY", '{"enable_thinking": false}')

    assert _load_settings().chat_extra_body == {"enable_thinking": False}


def test_chat_extra_body_rejects_a_non_object(monkeypatch) -> None:
    monkeypatch.setenv("RZYL_CHAT_EXTRA_BODY", "[1, 2]")

    with pytest.raises(ValidationError):
        _load_settings()


def test_loop_intervals_read_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("RZYL_WINDOW_FLUSH_SECONDS", "15")
    monkeypatch.setenv("RZYL_DEAD_LETTER_RETRY_SECONDS", "45")
    monkeypatch.setenv("RZYL_CHAT_TIMEOUT", "90")

    settings = _load_settings()

    assert settings.window_flush_seconds == 15
    assert settings.dead_letter_retry_seconds == 45
    assert settings.chat_timeout == 90.0
