"""Scheduling policy shared by the dependency-free planner and paid requests."""
import os
from datetime import timezone
from urllib.parse import urlparse


def off_peak_allowed(now):
    if now.tzinfo is None:
        raise ValueError('Off-peak checks require a timezone-aware datetime')
    now = now.astimezone(timezone.utc)
    minute = now.hour * 60 + now.minute
    return now.weekday() >= 5 or not (45 <= minute < 240 or 345 <= minute < 600)


def next_off_peak(now):
    if off_peak_allowed(now):
        return now
    now = now.astimezone(timezone.utc)
    return now.replace(hour=4 if now.hour < 4 else 10, minute=0, second=0, microsecond=0)


def scheduled_guard_enabled():
    if os.getenv('ENFORCE_DEEPSEEK_OFF_PEAK', '').lower() not in {'1', 'true', 'yes'}:
        return False
    if urlparse(os.getenv('OPENAI_BASE_URL', '').strip()).hostname != 'api.deepseek.com':
        return False
    default = os.getenv('MODEL_NAME', '').strip() or 'deepseek-flash'
    models = [os.getenv('DETAIL_MODEL_NAME', '').strip() or default]
    if os.getenv('SELECTION_MODE', 'semantic') == 'semantic' or os.getenv('USE_MODEL_FILTER', '').lower() in {'1', 'true', 'yes'}:
        models.append(os.getenv('FILTER_MODEL_NAME', '').strip() or default)
    if os.getenv('USE_MODEL_IMPORTANCE', '').lower() in {'1', 'true', 'yes'}:
        models.append(os.getenv('IMPORTANCE_MODEL_NAME', '').strip() or default)
    if os.getenv('ENABLE_DEEP_READ', 'true').lower() in {'1', 'true', 'yes'}:
        models.append(os.getenv('DEEP_READ_MODEL_NAME', '').strip() or default)
    return any(model.lower().startswith('deepseek-') for model in models)
