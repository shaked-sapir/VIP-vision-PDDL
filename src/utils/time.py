from datetime import datetime


def create_experiment_timestamp(time_format: str = "%d-%m-%YT%H:%M:%S") -> str:
    """Returns the current date and time formatted as needed, could be used for experiment naming but not only."""
    now = datetime.now()
    ts = now.strftime(time_format)
    return ts
