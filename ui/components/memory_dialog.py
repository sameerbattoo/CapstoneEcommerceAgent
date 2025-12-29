"""Memory dialog UI component."""

import streamlit as st
import re


def render_memory_dialog(memory_service, session_id: str, actor_id: str, aws_region: str):
    """Render memory dialog with tabs for conversation, preferences, and facts.
    
    Args:
        memory_service: MemoryService instance
        session_id: Session ID
        actor_id: Actor ID
        aws_region: AWS region
    """
    @st.dialog("Session Memory using AgentCore Memory", width="large")
    def show_memory():
        st.session_state.dialog_is_open = True
        
        # Initialize search state
        _init_search_state()
        
        st.write(f"**Session ID:** `{session_id}`")
        st.write(f"**Actor ID:** `{actor_id}`")
        
        # Load memory data
        with st.spinner("Loading session memory..."):
            memory_data = memory_service.fetch_session_memory(session_id, actor_id)
        
        if "error" in memory_data:
            st.error(f"❌ Error loading memory: {memory_data['error']}")
        else:
            # Create tabs
            tab1, tab2, tab3 = st.tabs([
                "💬 Conversation History",
                "⚙️ User Preferences",
                "📝 User Facts"
            ])
            
            with tab1:
                _render_conversation_tab(memory_data.get("turns", []))
            
            with tab2:
                _render_semantic_search_tab(
                    title="⚙️ User Preferences",
                    namespace="preferences",
                    default_results=memory_data.get("preferences", []),
                    default_query="What does the user prefer? What are their settings and product choices, preferred products?",
                    memory_service=memory_service,
                    actor_id=actor_id,
                    search_key_prefix="preferences"
                )
            
            with tab3:
                _render_semantic_search_tab(
                    title="📝 User Facts",
                    namespace="facts",
                    default_results=memory_data.get("facts", []),
                    default_query="What information do we know about the user? User email, location, past purchases, product reviews.",
                    memory_service=memory_service,
                    actor_id=actor_id,
                    search_key_prefix="facts"
                )
        
        if st.button("Close", type="primary", use_container_width=True):
            st.session_state.show_memory_dialog = False
            st.session_state.dialog_is_open = False
            st.rerun()
    
    show_memory()


def _init_search_state():
    """Initialize search state variables."""
    if "preferences_search_query" not in st.session_state:
        st.session_state.preferences_search_query = ""
    if "facts_search_query" not in st.session_state:
        st.session_state.facts_search_query = ""


def _render_conversation_tab(turns: list):
    """Render conversation history tab.
    
    Args:
        turns: List of conversation turns
    """
    st.subheader("💬 Conversation History")
    
    if turns:
        # Flatten all messages from all turns
        all_messages = []
        for turn in turns:
            for msg in turn:
                all_messages.append(msg)
        
        # Reverse to get chronological order (oldest first)
        all_messages.reverse()
        
        # Display messages
        turns_data = []
        for idx, msg in enumerate(all_messages, 1):
            role = msg.get("role", "unknown")
            content = msg.get("content", {}).get("text", "")
            turns_data.append({
                "#": idx,
                "Role": role,
                "Content": content[:1000] + "..." if len(content) > 1000 else content
            })
        
        if turns_data:
            st.dataframe(turns_data, use_container_width=True, hide_index=True)
            st.caption(f"📝 Showing {len(turns_data)} messages")
        else:
            st.info("No conversation history found")
    else:
        st.info("No conversation history found")


def _render_semantic_search_tab(
    title: str,
    namespace: str,
    default_results: list,
    default_query: str,
    memory_service,
    actor_id: str,
    search_key_prefix: str
):
    """Render a semantic search tab (reusable for preferences and facts).
    
    Args:
        title: Tab title
        namespace: Memory namespace (preferences or facts)
        default_results: Default results to show
        default_query: Default search query used
        memory_service: MemoryService instance
        actor_id: Actor ID
        search_key_prefix: Prefix for session state keys
    """
    st.subheader(title)
    
    # Search bar
    col1, col2 = st.columns([4, 1])
    with col1:
        search_query = st.text_input(
            f"Search {namespace}",
            placeholder=f"e.g., 'shipping', 'products', 'payment methods'...",
            key=f"{search_key_prefix}_search_input",
            label_visibility="collapsed"
        )
    with col2:
        search_button = st.button(
            "🔍 Search",
            key=f"{search_key_prefix}_search_btn",
            use_container_width=True
        )
    
    # Search controls
    col1, col2 = st.columns(2)
    with col1:
        relevance_threshold = st.slider(
            "Minimum Relevance",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            key=f"{search_key_prefix}_relevance"
        )
    with col2:
        top_k = st.selectbox(
            "Show results",
            options=[5, 10, 20, 50],
            index=1,
            key=f"{search_key_prefix}_top_k"
        )
    
    # Check if user has performed a search
    if search_button and search_query:
        st.session_state[f"{search_key_prefix}_search_query"] = search_query
    
    # Get the active search query
    active_query = st.session_state.get(f"{search_key_prefix}_search_query", "")
    
    # Decide which results to show
    if active_query:
        # User has searched - fetch new results
        results = _perform_search(
            memory_service,
            namespace,
            actor_id,
            active_query,
            top_k,
            relevance_threshold
        )
        st.info(f"🔍 Found {len(results)} {namespace} matching '{active_query}'")
    else:
        # No search - use default results
        results = default_results
        st.info(f"Showing top {len(results)} {namespace} (default search)")
        with st.expander("ℹ️ Default search query"):
            st.code(default_query)
    
    # Display results
    if results:
        _display_search_results(results, active_query)
        
        # Clear search button
        if active_query:
            if st.button("Clear Search", key=f"clear_{search_key_prefix}_search"):
                st.session_state[f"{search_key_prefix}_search_query"] = ""
                st.rerun()
    else:
        st.info(f"No {namespace} stored")


def _perform_search(
    memory_service,
    namespace: str,
    actor_id: str,
    query: str,
    top_k: int,
    relevance_threshold: float
) -> list:
    """Perform semantic search on memory.
    
    Args:
        memory_service: MemoryService instance
        namespace: Memory namespace
        actor_id: Actor ID
        query: Search query
        top_k: Number of results
        relevance_threshold: Minimum relevance score
        
    Returns:
        List of search results
    """
    with st.spinner(f"Searching {namespace} for '{query}'..."):
        if namespace == "preferences":
            results = memory_service.retrieve_preferences(actor_id, query, top_k)
        else:
            results = memory_service.retrieve_facts(actor_id, query, top_k)
        
        # Filter by relevance threshold
        return [r for r in results if r.get("score", 0) >= relevance_threshold]


def _display_search_results(results: list, active_query: str = ""):
    """Display search results in a table.
    
    Args:
        results: List of search results
        active_query: Active search query for highlighting
    """
    data = []
    for result in results:
        content = result.get("content", {}).get("text", "")
        score = result.get("score", 0)
        
        # Highlight search terms if user searched
        if active_query:
            pattern = re.compile(re.escape(active_query), re.IGNORECASE)
            content = pattern.sub(f"**{active_query}**", content)
        
        data.append({
            "Content": content,
            "Relevance": f"{score:.2f}"
        })
    
    st.dataframe(data, use_container_width=True, hide_index=True)
