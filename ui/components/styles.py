"""CSS styles for the application."""

import streamlit as st


def get_app_styles() -> str:
    """Return minimal CSS for better UI appearance."""
    return """
        <style>
            /* Improve button styling for sample questions */
            .stButton button {
                border-radius: 8px;
                border: 1px solid #e0e0e0;
                transition: all 0.2s ease;
                text-align: left;
                white-space: normal;
                height: auto;
                min-height: 40px;
                padding: 0.5rem 0.75rem;
            }
            
            .stButton button:hover {
                background: #f0f2f6;
                border-color: #FF4B4B;
                transform: translateY(-1px);
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            
            /* Improve expander styling */
            .streamlit-expanderHeader {
                font-weight: 600;
                font-size: 1rem;
            }
            
            /* Add some spacing */
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
            
            /* Dark mode support */
            @media (prefers-color-scheme: dark) {
                .stButton button {
                    background: #262730;
                    border-color: #4a4a4a;
                    color: #fafafa;
                }
                .stButton button:hover {
                    background: #1e1e1e;
                }
            }
        </style>
    """


def apply_app_styles():
    """Apply all application styles."""
    st.markdown(get_app_styles(), unsafe_allow_html=True)
