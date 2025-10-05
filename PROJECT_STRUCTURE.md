# Horizon AI Assistant - Project Structure

## Overview
Horizon has been restructured using a shared core architecture to support both web interface and MCP agent functionality.

## Architecture: Option A (Shared Core)

```
horizon/
├── core/                    # 🧠 Shared business logic
│   ├── __init__.py         # Core module initialization
│   ├── ai_engine.py        # ChatGPT/Gemini integration
│   ├── personality.py      # Personality system and blending
│   ├── database.py         # Database operations
│   ├── media_generator.py  # Image/video/audio generation
│   ├── memory_system.py    # User memory management
│   └── utils.py            # Shared utilities
│
├── web/                    # 🌐 Web interface (Flask)
│   ├── __init__.py         # Web module initialization
│   ├── app.py              # Flask application (refactored)
│   ├── routes.py           # Web route definitions
│   ├── templates/          # HTML templates (moved from root)
│   └── static/             # CSS/JS assets (moved from root)
│
├── mcp/                    # 🤖 MCP Agent (Protocol Server)
│   ├── __init__.py         # MCP module initialization
│   ├── server.py           # Main MCP server
│   ├── tools.py            # Tool implementations
│   ├── resources.py        # Resource handlers
│   ├── prompts.py          # Personality prompts
│   └── schemas.py          # Pydantic schemas
│
├── scripts/                # 🚀 Startup scripts
│   ├── __init__.py         # Scripts documentation
│   ├── start_web.py        # Start web server
│   ├── start_mcp.py        # Start MCP server
│   ├── start_both.py       # Start both services
│   └── dev_setup.py        # Development setup
│
├── tests/                  # 🧪 Test files (existing)
├── utils/                  # 🛠️ Utility modules (existing)
├── static/                 # 📁 Will be moved to web/static/
├── templates/              # 📁 Will be moved to web/templates/
├── app.py                  # 📁 Will be refactored to web/app.py
├── config.py               # Configuration (existing)
├── requirements.txt        # Dependencies (existing)
└── README.md               # Documentation (existing)
```

## Implementation Plan

### Week 1: Foundation & Architecture Setup (In Progress)
- [x] **Day 1**: Project structure creation ✅
- [x] **Day 1**: MCP dependency installation ✅
- [x] **Day 1**: MCP specification research ✅
- [x] **Day 1**: Development roadmap creation ✅
- [ ] **Day 2**: Core extraction from app.py
- [ ] **Day 3-4**: Refactor web interface
- [ ] **Day 5-7**: Basic MCP server setup

### Current Status: Day 1 Complete ✅

#### Completed:
1. ✅ Created new directory structure (`core/`, `mcp/`, `web/`, `scripts/`)
2. ✅ Installed MCP dependencies (`mcp`, `pydantic`, `uvicorn`, `starlette`)
3. ✅ Research MCP specification (Protocol version: 2025-06-18)
4. ✅ Created module initialization files with documentation
5. ✅ **Created comprehensive development roadmap** (see `DEVELOPMENT_ROADMAP.md`)

#### MCP Research Summary:
- **Protocol Version**: 2025-06-18 (current specification)
- **Architecture**: JSON-RPC based protocol for AI-to-AI communication
- **Components**: Tools, Resources, Prompts
- **Transport**: Standard input/output, HTTP, WebSocket support
- **Python Library**: `mcp` package with full protocol support

#### Next Steps (Day 2):
- Extract core AI functionality from `app.py`
- Create `core/ai_engine.py` with ChatGPT/Gemini integration
- Create `core/personality.py` with personality system
- Create `core/database.py` with database operations
- Create `core/media_generator.py` with generation functions

**📋 For detailed task breakdown and timeline, see**: [`DEVELOPMENT_ROADMAP.md`](./DEVELOPMENT_ROADMAP.md)

## Benefits of This Structure

### 🔄 Separation of Concerns
- **Core**: Pure business logic, no UI dependencies
- **Web**: HTTP handling, templates, user interface
- **MCP**: Protocol compliance, tool definitions, AI integration

### 🚀 Scalability
- Web and MCP can run independently or together
- Core logic is reusable across interfaces
- Easy to add new interfaces (CLI, API, etc.)

### 🧪 Testability
- Core logic can be tested in isolation
- Web and MCP interfaces can be tested separately
- Clear dependency boundaries

### 🔧 Maintainability
- Changes to core logic affect both interfaces
- Interface-specific changes don't affect core
- Clear module boundaries and responsibilities