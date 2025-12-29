# Refactored Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Application                     │
│              orch_web_app_cognito_refactored.py             │
│                        (200 lines)                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ orchestrates
                              ▼
        ┌─────────────────────────────────────────┐
        │                                         │
        ▼                                         ▼
┌──────────────┐                         ┌──────────────┐
│   Config     │                         │     Auth     │
│              │                         │              │
│  Settings    │                         │  UserAuth    │
│  Validation  │                         │  Session Mgr │
└──────────────┘                         └──────────────┘
        │                                         │
        │ provides config                         │ manages auth
        ▼                                         ▼
┌─────────────────────────────────────────────────────────────┐
│                         Services                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  AgentCore   │  │ Transcription│  │    Memory    │     │
│  │    Client    │  │   Service    │  │   Service    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
        │                     │                     │
        │ calls AWS           │ calls AWS           │ calls AWS
        ▼                     ▼                     ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   Bedrock    │      │  Transcribe  │      │  AgentCore   │
│  AgentCore   │      │      +       │      │    Memory    │
│              │      │     S3       │      │              │
└──────────────┘      └──────────────┘      └──────────────┘
        │
        │ streams responses
        ▼
┌─────────────────────────────────────────────────────────────┐
│                      UI Components                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Login Page  │  │   Sidebar    │  │     Chat     │     │
│  │              │  │              │  │  Interface   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│  ┌──────────────┐  ┌──────────────┐                       │
│  │    Memory    │  │    Styles    │                       │
│  │    Dialog    │  │     CSS      │                       │
│  └──────────────┘  └──────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

## Module Dependencies

```
orch_web_app_cognito_refactored.py
    │
    ├── config/
    │   └── settings.py
    │       └── (loads .env)
    │
    ├── auth/
    │   ├── cognito_auth.py
    │   │   └── boto3.cognito-idp
    │   └── session_manager.py
    │       └── pickle (for persistence)
    │
    ├── services/
    │   ├── agentcore_client.py
    │   │   ├── boto3.bedrock-agentcore
    │   │   └── requests (for streaming)
    │   ├── transcription_service.py
    │   │   ├── boto3.transcribe
    │   │   └── boto3.s3
    │   └── memory_service.py
    │       └── bedrock_agentcore.memory
    │
    ├── components/
    │   ├── login_page.py
    │   │   ├── auth.UserAuth
    │   │   └── auth.SessionManager
    │   ├── sidebar.py
    │   │   └── services.TranscriptionService
    │   ├── chat_interface.py
    │   │   └── services.AgentCoreClient
    │   ├── memory_dialog.py
    │   │   └── services.MemoryService
    │   └── styles.py
    │       └── (pure CSS)
    │
    └── utils/
        └── helpers.py
            └── (pure functions)
```

## Data Flow

### 1. User Login Flow
```
User enters credentials
        │
        ▼
render_login_page()
        │
        ▼
UserAuth.authenticate()
        │
        ▼
AWS Cognito
        │
        ▼
SessionManager.save_session()
        │
        ▼
Session State Updated
        │
        ▼
Main App Rendered
```

### 2. Chat Message Flow
```
User types message
        │
        ▼
render_chat_input()
        │
        ▼
process_agent_response()
        │
        ▼
AgentCoreClient.invoke_streaming()
        │
        ▼
AWS Bedrock AgentCore
        │
        ▼
Stream events back
        │
        ├── [THINKING] → thinking_placeholder
        ├── [TOOL USE] → tool_use_placeholder
        └── content → answer_placeholder
        │
        ▼
Display in chat
```

### 3. Voice Input Flow
```
User records audio
        │
        ▼
render_sidebar() → voice input
        │
        ▼
TranscriptionService.transcribe_audio()
        │
        ├── Upload to S3
        ├── Start Transcribe job
        ├── Wait for completion
        └── Get transcript
        │
        ▼
Add to messages
        │
        ▼
Process as chat message
```

### 4. Memory Dialog Flow
```
User clicks session ID
        │
        ▼
render_memory_dialog()
        │
        ▼
MemoryService.fetch_session_memory()
        │
        ├── get_last_k_turns()
        ├── retrieve_preferences()
        └── retrieve_facts()
        │
        ▼
Display in tabs
        │
        ├── Conversation History
        ├── User Preferences (with search)
        └── User Facts (with search)
```

## Component Interaction

```
┌─────────────────────────────────────────────────────────────┐
│                         Main App                             │
│                                                              │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐       │
│  │  Settings  │───▶│  Services  │───▶│    AWS     │       │
│  └────────────┘    └────────────┘    └────────────┘       │
│         │                  │                                │
│         │                  │                                │
│         ▼                  ▼                                │
│  ┌────────────┐    ┌────────────┐                         │
│  │    Auth    │    │ Components │                         │
│  └────────────┘    └────────────┘                         │
│         │                  │                                │
│         └──────────┬───────┘                                │
│                    │                                        │
│                    ▼                                        │
│            ┌────────────────┐                              │
│            │ Session State  │                              │
│            └────────────────┘                              │
└─────────────────────────────────────────────────────────────┘
```

## Separation of Concerns

### Layer 1: Configuration
- **Responsibility**: Load and validate environment variables
- **Files**: `config/settings.py`
- **Dependencies**: `python-dotenv`

### Layer 2: Authentication
- **Responsibility**: Handle user authentication and session management
- **Files**: `auth/cognito_auth.py`, `auth/session_manager.py`
- **Dependencies**: `boto3`, `pickle`

### Layer 3: Services
- **Responsibility**: Interact with AWS services
- **Files**: `services/*.py`
- **Dependencies**: `boto3`, `requests`, `bedrock_agentcore`

### Layer 4: UI Components
- **Responsibility**: Render user interface
- **Files**: `components/*.py`
- **Dependencies**: `streamlit`, services layer

### Layer 5: Main Application
- **Responsibility**: Orchestrate all layers
- **Files**: `orch_web_app_cognito_refactored.py`
- **Dependencies**: All layers

## Error Handling Flow

```
Exception occurs in service
        │
        ▼
Custom exception raised
(AgentCoreError, TranscriptionError, etc.)
        │
        ▼
Caught in component
        │
        ├── UnauthorizedError → Clear session, force re-login
        ├── TranscriptionError → Show error, allow retry
        └── AgentCoreError → Show error, log details
        │
        ▼
User-friendly message displayed
```

## State Management

```
st.session_state
    │
    ├── authenticated (bool)
    ├── username (str)
    ├── id_token (str)
    ├── access_token (str)
    ├── refresh_token (str)
    ├── tenant_id (str)
    │
    ├── messages (list)
    ├── session_id (str)
    │
    ├── challenge_name (str | None)
    ├── challenge_session (str | None)
    ├── temp_username (str | None)
    │
    ├── show_memory_dialog (bool)
    ├── dialog_is_open (bool)
    │
    ├── preferences_search_query (str)
    └── facts_search_query (str)
```

## Key Design Patterns

### 1. Service Layer Pattern
- Services encapsulate AWS interactions
- Clean interfaces for business logic
- Easy to mock for testing

### 2. Component Pattern
- UI components are reusable
- Props-based configuration
- Separation from business logic

### 3. Configuration Pattern
- Centralized configuration
- Validation on startup
- Type-safe with dataclasses

### 4. Session Management Pattern
- Persistent sessions across restarts
- Automatic expiration
- Secure token storage

### 5. Error Handling Pattern
- Custom exceptions for different error types
- Consistent error handling across layers
- User-friendly error messages

## Benefits of This Architecture

✅ **Modularity**: Each module has a single, clear responsibility  
✅ **Testability**: Services and components can be tested in isolation  
✅ **Maintainability**: Easy to locate and modify specific functionality  
✅ **Scalability**: Simple to add new features without affecting existing code  
✅ **Reusability**: Components and services can be used in other applications  
✅ **Type Safety**: Type hints throughout improve IDE support and catch errors  
✅ **Error Handling**: Consistent patterns make debugging easier  
✅ **Team Collaboration**: Multiple developers can work on different modules  

## Comparison to Original

### Original Architecture
```
┌─────────────────────────────────────┐
│                                     │
│   orch_web_app_cognito.py          │
│         (1497 lines)                │
│                                     │
│  Everything in one file:            │
│  - Config                           │
│  - Auth                             │
│  - Services                         │
│  - UI                               │
│  - Styles                           │
│  - Utils                            │
│                                     │
└─────────────────────────────────────┘
```

### Refactored Architecture
```
┌─────────────────────────────────────┐
│  orch_web_app_cognito_refactored   │
│         (200 lines)                 │
└─────────────────────────────────────┘
         │
         ├── config/
         ├── auth/
         ├── services/
         ├── components/
         └── utils/
```

**Result**: Clean, maintainable, professional architecture! 🎉
