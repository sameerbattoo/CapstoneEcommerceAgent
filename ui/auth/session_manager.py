"""Session persistence management."""

import pickle
import time
from pathlib import Path


class SessionManager:
    """Manages authentication session persistence."""
    
    SESSION_EXPIRY_HOURS = 24
    SESSION_EXPIRY_SECONDS = SESSION_EXPIRY_HOURS * 3600
    
    @staticmethod
    def get_session_file() -> Path:
        """Get the path to the session file."""
        session_dir = Path.home() / ".streamlit_sessions"
        session_dir.mkdir(exist_ok=True)
        return session_dir / "auth_session.pkl"
    
    @classmethod
    def save_session(cls, username: str, id_token: str, access_token: str, refresh_token: str):
        """Save authentication session to disk.
        
        Args:
            username: Username
            id_token: ID token
            access_token: Access token
            refresh_token: Refresh token
        """
        try:
            session_data = {
                "username": username,
                "id_token": id_token,
                "access_token": access_token,
                "refresh_token": refresh_token,
                "timestamp": time.time()
            }
            with open(cls.get_session_file(), "wb") as f:
                pickle.dump(session_data, f)
        except Exception:
            # Silently fail - session persistence is not critical
            pass
    
    @classmethod
    def load_session(cls) -> dict | None:
        """Load authentication session from disk.
        
        Returns:
            Session data dictionary or None if not found/expired
        """
        try:
            session_file = cls.get_session_file()
            if session_file.exists():
                with open(session_file, "rb") as f:
                    session_data = pickle.load(f)
                
                # Check if session is still valid
                if time.time() - session_data.get("timestamp", 0) < cls.SESSION_EXPIRY_SECONDS:
                    return session_data
                else:
                    # Session expired, delete it
                    session_file.unlink()
        except Exception:
            pass
        return None
    
    @classmethod
    def clear_session(cls):
        """Clear authentication session from disk."""
        try:
            session_file = cls.get_session_file()
            if session_file.exists():
                session_file.unlink()
        except Exception:
            pass
