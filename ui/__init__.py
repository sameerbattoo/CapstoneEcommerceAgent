"""UI interfaces package."""

from .orch_web_app_local import main as run_web_local
from .orch_web_app_iam import main as run_web_iam
from .orch_web_app_cognito import main as run_web_cognito

__all__ = ["run_web_local", "run_web_iam", "run_web_cognito"]
