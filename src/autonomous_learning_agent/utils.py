from datetime import datetime


def get_today_str() -> str:
    """Get current date in human readable format"""
    return datetime.now().strftime("%a %b %#d,%Y")
