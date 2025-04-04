# Agent Village: Multi-Agent LLM Collaboration System

![Agent Village Logo](https://example.com/agent_village_logo.png)

## 🌟 Overview

Agent Village is a comprehensive framework for creating collaborative multi-agent LLM systems. It enables multiple AI agents to work together in a shared environment, access virtual computers, utilize tools, and collaborate toward common goals.

Unlike traditional single-agent systems, Agent Village provides a structured environment where specialized agents can communicate, share knowledge, and work together to solve complex problems—similar to human teams. The system includes persistent memory, virtual computers for code execution, and a project management structure to coordinate activities.

## 🏛️ Architecture

Agent Village consists of six core modules:

1. **Core Agent Framework**: Foundation for creating specialized intelligent agents with perception, cognition, and action capabilities.
2. **Chatroom System**: Shared communication environment for agent interaction.
3. **Virtual Computer System**: Secure, sandboxed environments for executing code and manipulating files.
4. **Offline Memory Database**: Persistent storage for agent knowledge and experiences.
5. **Orchestration System**: High-level coordination for running the agent village.
6. **API and Frontend Interface**: Web interface for monitoring and interacting with the system.

![Architecture Diagram](https://example.com/architecture_diagram.png)

## 🚀 Installation

### Prerequisites

- Python 3.10+
- Docker (for sandboxed code execution)
- SQLite (for database storage)
- Modern web browser (for UI)

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/agent-village.git
cd agent-village

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p agent_village_data/chatroom
mkdir -p agent_village_data/memory
mkdir -p agent_village_data/computers/main
mkdir -p agent_village_data/computers/data
mkdir -p templates

# Configure environment variables
cp .env.example .env
# Edit .env with your API keys and settings
```

## ⚙️ Configuration

Agent Village is configured using JSON files. A default configuration is provided in `config/agent_village.json`:

```bash
# Create configuration directory
mkdir -p config

# Copy example configuration
cp examples/agent_village.json config/
```

Edit the configuration file to customize:
- Agent roles and capabilities
- Available virtual computers
- Memory system parameters
- Chatroom settings

## 🔌 Usage

### Starting the System

```bash
# Start the Agent Village API server
python -m api

# Access the web interface
# Open http://localhost:8000 in your browser
```

### Programmatic Usage

```python
from orchestration import AgentVillage, OrchestrationConfig
import asyncio
import json

async def run_agent_village():
    # Load configuration
    with open("config/agent_village.json", "r") as f:
        config_data = json.load(f)
    
    # Create orchestration config
    orchestration_config = OrchestrationConfig(**config_data)
    
    # Initialize Agent Village
    village = AgentVillage(orchestration_config)
    
    # Start the village
    await village.start()
    
    # Interact with the village
    await village.send_message(
        from_name="User",
        to_name="ProjectManager",
        content="Let's improve the codebase",
        message_type="chat"
    )
    
    # Keep the village running
    try:
        # Run for a set time or until interrupted
        await asyncio.sleep(3600)  # Run for 1 hour
    finally:
        # Gracefully shut down
        await village.stop()

# Run the agent village
asyncio.run(run_agent_village())
```

## 🧠 Core Concepts

### Agents

Agents are autonomous entities that can perceive, think, and act within the village. Each agent has:
- A specialized role (e.g., Project Manager, Software Engineer)
- Access to specific tools
- Memory for storing experiences
- Communication capabilities

### Tools

Tools enable agents to interact with the environment:
- File operations (read, write, delete)
- Code execution (Python, JavaScript, Bash)
- Project management
- Data analysis

### Projects

Projects provide structure for agent collaboration:
- Defined goals and tasks
- Team of specialized agents
- Resources and artifacts
- Progress tracking

## 📚 API Documentation

The Agent Village exposes a RESTful API for integration with other systems:

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | Get system status |
| `/api/message` | POST | Send a message |
| `/api/command` | POST | Execute a command |
| `/api/agents` | GET | List all agents |
| `/api/agents` | POST | Add a new agent |
| `/api/agents/{name}` | DELETE | Remove an agent |
| `/api/projects` | GET | List all projects |
| `/api/projects` | POST | Create a new project |
| `/api/memories/{agent_name}` | GET | Get agent memories |
| `/api/search/memories` | GET | Search agent memories |

### WebSocket

Real-time updates are available via WebSocket:
- Connect to `/ws` to receive events
- Events include new messages, agent status changes, and command results

## 🛠️ Development

### Project Structure

```
agent-village/
├── agent_framework.py  # Core agent implementation
├── chatroom.py         # Communication environment
├── virtual_computer.py # Tool execution environment
├── memory_db.py        # Persistent memory system
├── orchestration.py    # System coordination
├── api.py              # API and web interface
├── templates/          # HTML templates
│   └── index.html      # Main UI
├── config/             # Configuration files
│   └── agent_village.json
├── requirements.txt    # Dependencies
└── README.md           # This file
```

### Running Tests

```bash
# Run unit tests
python -m unittest discover tests

# Run integration tests
python -m unittest discover integration_tests
```

### Adding New Agent Types

1. Extend the `Agent` class in `agent_framework.py`
2. Implement the abstract methods `think()` and `decide_action()`
3. Register the new agent type in `create_agent()` factory function

### Adding New Tools

1. Create a new tool function in the appropriate module
2. Create a `Tool` object with the function and metadata
3. Add the tool to the available tools in `VirtualComputer`

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please adhere to the coding standards and add tests for new features.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔍 Troubleshooting

### Common Issues

1. **WebSocket Connection Failed**
   - Ensure the API server is running
   - Check network connectivity and firewall settings

2. **Agent Creation Failed**
   - Verify configuration format
   - Ensure LLM API keys are properly set

3. **Code Execution Timeout**
   - Increase `max_execution_time` in computer configuration
   - Simplify the code being executed

4. **Memory Database Errors**
   - Check storage permissions
   - Ensure database paths are correctly configured

### Logs

Logs are stored in the project root:
- `agent_village.log`: General system logs
- `api.log`: API server logs
- `orchestration.log`: Orchestration system logs
- `memory_db.log`: Memory database logs
- `virtual_computer.log`: Virtual computer logs

## 🙏 Acknowledgements

- The OpenAI and Anthropic teams for their language model research
- The LangChain project for inspiration
- All contributors and community members

---

Created with ❤️ by [Your Name]
