from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    blink_username: str
    blink_password: str

    cameras: list[str] = Field(default_factory=list)

    poll_interval_seconds: int = 20

    max_concurrent_downloads: int = 2

    video_dir: Path = Path("./videos")
    max_disk_gb: float = 10.0

    state_dir: Path = Path("./data")
    session_file: Path = Path("./data/blink_session.json")

    timezone: str = "America/Los_Angeles"

    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None

    gemini_api_key: Optional[str] = None
    gemini_video_model: Optional[str] = None
    gemini_chat_model: Optional[str] = None
    chat_history_dir: Path = Path("./data/chat_history")
    events_dir: Path = Path("./events")

    @field_validator("cameras", mode="before")
    @classmethod
    def _split_cameras(cls, v):
        if isinstance(v, str):
            return [c.strip() for c in v.split(",") if c.strip()]
        return v

    def ensure_dirs(self) -> None:
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)


def load_settings() -> Settings:
    s = Settings()
    if not s.cameras:
        raise SystemExit("CAMERAS must list at least one camera name — refusing to watch all.")
    s.ensure_dirs()
    return s
