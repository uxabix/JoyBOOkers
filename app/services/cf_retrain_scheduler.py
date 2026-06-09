"""Background CF retrain triggered after enough in-app rating changes."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from sqlalchemy.orm import Session

from app.config import Settings
from app.db.session import get_session_factory
from app.logging_config import get_logger
from app.ml.cf_retrain import run_cf_retrain

if TYPE_CHECKING:
    from app.ml.registry import MLModelRegistry

logger = get_logger(__name__)


class CfRetrainScheduler:
    def __init__(
        self,
        settings: Settings,
        *,
        registry_getter: Callable[[], MLModelRegistry | None] | None = None,
        retrain_fn: Callable[[Settings, Session], dict[str, Any]] | None = None,
    ) -> None:
        self.settings = settings
        self._registry_getter = registry_getter or (lambda: None)
        self._retrain_fn = retrain_fn or run_cf_retrain
        self._lock = threading.Lock()
        self._dirty_count = 0
        self._retraining = False
        self._last_retrain_at = 0.0
        self._last_report: dict[str, Any] | None = None

    def record_app_change(self) -> None:
        if not self.settings.cf_retrain_enabled:
            return

        with self._lock:
            self._dirty_count += 1
            count = self._dirty_count
            threshold = self.settings.cf_retrain_threshold

        logger.debug("CF retrain dirty count: %s/%s", count, threshold)
        if count >= threshold:
            self._maybe_schedule_retrain()

    def _maybe_schedule_retrain(self) -> None:
        with self._lock:
            if self._retraining:
                return
            if self._dirty_count < self.settings.cf_retrain_threshold:
                return
            elapsed = time.monotonic() - self._last_retrain_at
            if elapsed < self.settings.cf_retrain_min_interval_seconds:
                return
            self._retraining = True

        thread = threading.Thread(target=self._retrain_worker, name="cf-retrain", daemon=True)
        thread.start()

    def _retrain_worker(self) -> None:
        try:
            logger.info("CF retrain starting (threshold=%s)", self.settings.cf_retrain_threshold)
            Session = get_session_factory()
            with Session() as session:
                report = self._retrain_fn(self.settings, session)

            registry = self._registry_getter()
            if registry is not None:
                registry.reload_cf()

            with self._lock:
                self._last_report = report

            logger.info(
                "CF retrain complete: %s app ratings, RMSE=%.4f",
                report.get("app_ratings_exported", 0),
                report.get("validation_rmse", 0.0),
            )
        except Exception:
            logger.exception("CF retrain failed")
        finally:
            with self._lock:
                self._retraining = False
                self._dirty_count = 0
                self._last_retrain_at = time.monotonic()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": self.settings.cf_retrain_enabled,
                "threshold": self.settings.cf_retrain_threshold,
                "dirty_count": self._dirty_count,
                "retraining": self._retraining,
                "last_report": self._last_report,
            }
