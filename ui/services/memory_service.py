"""AWS Bedrock AgentCore Memory service."""

from bedrock_agentcore.memory import MemoryClient


class MemoryService:
    """Service for interacting with AgentCore Memory."""
    
    DEFAULT_TOP_K = 10
    
    def __init__(self, region: str, memory_id: str):
        """Initialize memory service.
        
        Args:
            region: AWS region
            memory_id: AgentCore memory ID
        """
        self.region = region
        self.memory_id = memory_id
        self.client = MemoryClient(region_name=region)
    
    def fetch_session_memory(self, session_id: str, actor_id: str) -> dict:
        """Fetch stored memory for a session.
        
        Args:
            session_id: Session ID
            actor_id: Actor ID
            
        Returns:
            Dictionary with turns, preferences, and facts
        """
        try:
            # Get last conversation turns
            turns = self.client.get_last_k_turns(
                memory_id=self.memory_id,
                actor_id=actor_id,
                session_id=session_id,
                k=10
            )
            
            # Get user preferences
            preferences = self.retrieve_preferences(
                actor_id=actor_id,
                query="What does the user prefer? What are their settings and product choices, preferred products?",
                top_k=self.DEFAULT_TOP_K
            )
            
            # Get user facts
            facts = self.retrieve_facts(
                actor_id=actor_id,
                query="What information do we know about the user? User email, location, past purchases, product reviews.",
                top_k=self.DEFAULT_TOP_K
            )
            
            return {
                "turns": turns,
                "preferences": preferences,
                "facts": facts
            }
        except Exception as e:
            return {"error": str(e)}
    
    def retrieve_preferences(self, actor_id: str, query: str, top_k: int = DEFAULT_TOP_K) -> list:
        """Retrieve user preferences using semantic search.
        
        Args:
            actor_id: Actor ID
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of preference memories
        """
        return self.client.retrieve_memories(
            memory_id=self.memory_id,
            namespace=f"/users/{actor_id}/preferences",
            query=query,
            top_k=top_k
        )
    
    def retrieve_facts(self, actor_id: str, query: str, top_k: int = DEFAULT_TOP_K) -> list:
        """Retrieve user facts using semantic search.
        
        Args:
            actor_id: Actor ID
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of fact memories
        """
        return self.client.retrieve_memories(
            memory_id=self.memory_id,
            namespace=f"/users/{actor_id}/facts",
            query=query,
            top_k=top_k
        )


class MemoryError(Exception):
    """Exception for memory service errors."""
    pass
