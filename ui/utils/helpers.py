"""Helper utility functions."""


def get_actor_id(username: str) -> str:
    """Convert username to actor ID format.
    
    Args:
        username: Username
        
    Returns:
        Actor ID in email format with special characters replaced
    """
    return f"{username}@email.com".replace('.', '_').replace('@', '-')
