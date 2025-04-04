"""
API and Frontend Interface Module - Web interface for the Agent Village

This module provides a web-based API and user interface for interacting with
the Agent Village system.

Dependencies:
- Python 3.10+
- fastapi 0.95.0+
- uvicorn 0.21.1+
- jinja2 3.1.2+
- aiohttp 3.8.4+
- pydantic 2.3+
"""

import asyncio
import datetime
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable

import aiohttp
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException, Depends, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from orchestration import AgentVillage, OrchestrationConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# API Models
class MessageRequest(BaseModel):
    """Request model for sending a message"""
    from_name: str
    to_name: Optional[str] = None
    content: str
    message_type: str = "chat"


class CommandRequest(BaseModel):
    """Request model for executing a command"""
    computer_name: str
    tool_name: str
    args: List[Any] = Field(default_factory=list)
    kwargs: Dict[str, Any] = Field(default_factory=dict)
    agent_name: Optional[str] = None


class NewAgentRequest(BaseModel):
    """Request model for creating a new agent"""
    name: str
    role: str
    description: str
    llm_model: str = "gpt-4"
    tools: List[str] = Field(default_factory=list)
    permissions: List[str] = Field(default_factory=list)
    system_prompt: Optional[str] = None


class NewProjectRequest(BaseModel):
    """Request model for creating a new project"""
    name: str
    description: str
    goals: List[str]
    team_members: Optional[List[str]] = None


# Initialize FastAPI
app = FastAPI(
    title="Agent Village API",
    description="API for interacting with the Agent Village collaborative AI system",
    version="1.0.0"
)

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Global reference to the Agent Village
agent_village: Optional[AgentVillage] = None

# Active WebSocket connections
active_connections: List[WebSocket] = []


async def get_agent_village() -> AgentVillage:
    """
    Dependency for getting the Agent Village instance
    
    Returns:
        The Agent Village instance
    """
    if agent_village is None:
        raise HTTPException(status_code=503, detail="Agent Village not initialized")
    return agent_village


@app.on_event("startup")
async def startup_event():
    """Initialize the Agent Village on startup"""
    global agent_village
    
    try:
        # Load configuration
        config_path = os.getenv("AGENT_VILLAGE_CONFIG", "config/agent_village.json")
        
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_data = json.load(f)
            
            # Create orchestration config
            orchestration_config = OrchestrationConfig(**config_data)
            
            # Initialize Agent Village
            agent_village = AgentVillage(orchestration_config)
            
            # Start the village
            await agent_village.start()
            
            logger.info("Agent Village initialized and started")
        else:
            logger.error(f"Configuration file not found: {config_path}")
    except Exception as e:
        logger.error(f"Error initializing Agent Village: {str(e)}")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown the Agent Village on application shutdown"""
    global agent_village
    
    if agent_village:
        try:
            await agent_village.stop()
            logger.info("Agent Village stopped")
        except Exception as e:
            logger.error(f"Error stopping Agent Village: {str(e)}")


# WebSocket for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()
            
            # Process message
            try:
                message_data = json.loads(data)
                message_type = message_data.get("type", "unknown")
                
                if message_type == "ping":
                    # Respond to ping
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": time.time()
                    })
                elif message_type == "subscribe":
                    # Handle subscription to topics
                    topics = message_data.get("topics", [])
                    await websocket.send_json({
                        "type": "subscription_ack",
                        "topics": topics,
                        "timestamp": time.time()
                    })
                else:
                    # Unknown message type
                    await websocket.send_json({
                        "type": "error",
                        "error": f"Unknown message type: {message_type}",
                        "timestamp": time.time()
                    })
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "error": "Invalid JSON data",
                    "timestamp": time.time()
                })
    except WebSocketDisconnect:
        active_connections.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        if websocket in active_connections:
            active_connections.remove(websocket)


async def broadcast_to_websockets(message: Dict[str, Any]):
    """
    Broadcast a message to all active WebSocket connections
    
    Args:
        message: Message to broadcast
    """
    # Add timestamp to message
    message["timestamp"] = time.time()
    
    # Broadcast to all connections
    for connection in active_connections:
        try:
            await connection.send_json(message)
        except Exception as e:
            logger.error(f"Error broadcasting to WebSocket: {str(e)}")
            # Remove failed connection
            if connection in active_connections:
                active_connections.remove(connection)


# API Routes

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """Render the main UI page"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/status")
async def get_status(village: AgentVillage = Depends(get_agent_village)):
    """Get the current status of the Agent Village"""
    return village.get_status()


@app.post("/api/message")
async def send_message(
    message_request: MessageRequest,
    village: AgentVillage = Depends(get_agent_village)
):
    """Send a message in the Agent Village"""
    result = await village.send_message(
        from_name=message_request.from_name,
        to_name=message_request.to_name,
        content=message_request.content,
        message_type=message_request.message_type
    )
    
    # Broadcast message event if successful
    if result.get("success"):
        await broadcast_to_websockets({
            "type": "new_message",
            "message": {
                "id": result["message_id"],
                "sender": message_request.from_name,
                "recipient": message_request.to_name,
                "content": message_request.content,
                "message_type": message_request.message_type,
                "timestamp": result["timestamp"]
            }
        })
    
    return result


@app.post("/api/command")
async def execute_command(
    command_request: CommandRequest,
    village: AgentVillage = Depends(get_agent_village)
):
    """Execute a command on a virtual computer"""
    result = await village.execute_computer_command(
        computer_name=command_request.computer_name,
        tool_name=command_request.tool_name,
        args=command_request.args,
        kwargs=command_request.kwargs,
        agent_name=command_request.agent_name
    )
    
    # Broadcast command event if successful
    if result.get("success"):
        await broadcast_to_websockets({
            "type": "command_executed",
            "command": {
                "computer": command_request.computer_name,
                "tool": command_request.tool_name,
                "agent": command_request.agent_name,
                "success": True
            }
        })
    
    return result


@app.get("/api/messages")
async def get_messages(
    limit: int = Query(50, ge=1, le=500),
    agent_name: Optional[str] = None,
    village: AgentVillage = Depends(get_agent_village)
):
    """Get recent messages"""
    return village.get_recent_messages(limit=limit, agent_name=agent_name)


@app.get("/api/agents")
async def get_agents(village: AgentVillage = Depends(get_agent_village)):
    """Get all agents in the village"""
    status = village.get_status()
    return {
        "agents": status["agents"],
        "count": status["agents"]["total"]
    }


@app.post("/api/agents")
async def add_agent(
    agent_request: NewAgentRequest,
    village: AgentVillage = Depends(get_agent_village)
):
    """Add a new agent to the village"""
    config = {
        "name": agent_request.name,
        "role": agent_request.role,
        "description": agent_request.description,
        "llm_model": agent_request.llm_model,
        "tools": agent_request.tools,
        "permissions": agent_request.permissions
    }
    
    if agent_request.system_prompt:
        config["system_prompt"] = agent_request.system_prompt
    
    result = await village.add_agent(config)
    
    # Broadcast agent added event if successful
    if result.get("success"):
        await broadcast_to_websockets({
            "type": "agent_added",
            "agent": {
                "name": agent_request.name,
                "role": agent_request.role,
                "id": result["agent_id"]
            }
        })
    
    return result


@app.delete("/api/agents/{name}")
async def remove_agent(
    name: str,
    village: AgentVillage = Depends(get_agent_village)
):
    """Remove an agent from the village"""
    result = await village.remove_agent(name)
    
    # Broadcast agent removed event if successful
    if result.get("success"):
        await broadcast_to_websockets({
            "type": "agent_removed",
            "agent": {
                "name": name
            }
        })
    
    return result


@app.post("/api/projects")
async def create_project(
    project_request: NewProjectRequest,
    village: AgentVillage = Depends(get_agent_village)
):
    """Create a new project in the village"""
    result = await village.create_project(
        name=project_request.name,
        description=project_request.description,
        goals=project_request.goals,
        team_members=project_request.team_members
    )
    
    # Broadcast project created event if successful
    if result.get("success"):
        await broadcast_to_websockets({
            "type": "project_created",
            "project": {
                "id": result["project_id"],
                "name": project_request.name,
                "team_members": result["team_members"]
            }
        })
    
    return result


@app.get("/api/projects")
async def get_projects(village: AgentVillage = Depends(get_agent_village)):
    """Get all projects in the village"""
    status = village.get_status()
    return {
        "projects": status["projects"],
        "count": status["projects"]["total"]
    }




@app.get("/api/memories/{agent_name}")
async def get_agent_memories(
    agent_name: str,
    memory_type: Optional[str] = None,
    limit: int = Query(20, ge=1, le=100),
    village: AgentVillage = Depends(get_agent_village)
):
    """Get memories for an agent"""
    return village.get_agent_memories(
        agent_name=agent_name,
        memory_type=memory_type,
        limit=limit
    )


@app.get("/api/search/memories")
async def search_memories(
    query: str,
    agent_name: Optional[str] = None,
    limit: int = Query(10, ge=1, le=50),
    village: AgentVillage = Depends(get_agent_village)
):
    """Search agent memories"""
    return village.search_memories(
        query=query,
        agent_name=agent_name,
        limit=limit
    )


@app.get("/api/computers")
async def get_computers(village: AgentVillage = Depends(get_agent_village)):
    """Get all virtual computers in the village"""
    return {
        "computers": [
            {
                "name": name,
                "tools": computer.get_available_tools()
            }
            for name, computer in village.virtual_computers.items()
        ],
        "count": len(village.virtual_computers)
    }


@app.get("/api/computers/{name}/tools")
async def get_computer_tools(
    name: str,
    village: AgentVillage = Depends(get_agent_village)
):
    """Get available tools for a virtual computer"""
    if name not in village.virtual_computers:
        raise HTTPException(status_code=404, detail=f"Computer {name} not found")
    
    computer = village.virtual_computers[name]
    return {
        "computer": name,
        "tools": computer.get_available_tools()
    }


@app.get("/api/virtual-space/rooms")
async def get_rooms(village: AgentVillage = Depends(get_agent_village)):
    """Get all rooms in the virtual space"""
    return {
        "rooms": [
            {
                "id": room_id,
                "name": room["name"],
                "description": room["description"],
                "agents": list(room["agents"])
            }
            for room_id, room in village.virtual_space.rooms.items()
        ],
        "count": len(village.virtual_space.rooms)
    }


@app.post("/api/virtual-space/rooms/{room_id}/move/{agent_name}")
async def move_agent_to_room(
    room_id: str,
    agent_name: str,
    village: AgentVillage = Depends(get_agent_village)
):
    """Move an agent to a different room"""
    try:
        await village.virtual_space.move_agent_to_room(agent_name, room_id)
        
        # Broadcast agent moved event
        await broadcast_to_websockets({
            "type": "agent_moved",
            "agent": agent_name,
            "room": room_id,
            "room_name": village.virtual_space.rooms[room_id]["name"]
        })
        
        return {
            "success": True,
            "agent": agent_name,
            "room": room_id,
            "room_name": village.virtual_space.rooms[room_id]["name"]
        }
    except Exception as e:
        logger.error(f"Error moving agent {agent_name} to room {room_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/virtual-space/workspaces")
async def get_workspaces(village: AgentVillage = Depends(get_agent_village)):
    """Get all workspaces in the virtual space"""
    return {
        "workspaces": [
            {
                "id": workspace_id,
                "name": workspace["name"],
                "description": workspace["description"],
                "owner": workspace["owner"],
                "collaborators": list(workspace["collaborators"]),
                "documents": len(workspace["documents"])
            }
            for workspace_id, workspace in village.virtual_space.workspaces.items()
        ],
        "count": len(village.virtual_space.workspaces)
    }


@app.get("/api/virtual-space/workspaces/{workspace_id}/documents")
async def get_workspace_documents(
    workspace_id: str,
    village: AgentVillage = Depends(get_agent_village)
):
    """Get documents in a workspace"""
    if workspace_id not in village.virtual_space.workspaces:
        raise HTTPException(status_code=404, detail=f"Workspace {workspace_id} not found")
    
    workspace = village.virtual_space.workspaces[workspace_id]
    return {
        "workspace": {
            "id": workspace_id,
            "name": workspace["name"],
            "description": workspace["description"]
        },
        "documents": [
            {
                "id": doc_id,
                "title": doc["title"],
                "author": doc["author"],
                "created_at": doc["created_at"],
                "updated_at": doc["updated_at"],
                "version": doc["version"]
            }
            for doc_id, doc in workspace["documents"].items()
        ],
        "count": len(workspace["documents"])
    }


@app.get("/api/virtual-space/workspaces/{workspace_id}/documents/{document_id}")
async def get_document(
    workspace_id: str,
    document_id: str,
    village: AgentVillage = Depends(get_agent_village)
):
    """Get a document from a workspace"""
    if workspace_id not in village.virtual_space.workspaces:
        raise HTTPException(status_code=404, detail=f"Workspace {workspace_id} not found")
    
    document = village.virtual_space.get_document(workspace_id, document_id)
    if not document:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    
    return {
        "document": {
            "id": document_id,
            "title": document["title"],
            "content": document["content"],
            "author": document["author"],
            "created_at": document["created_at"],
            "updated_at": document["updated_at"],
            "version": document["version"]
        }
    }


# Main entry point for running the API server
def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the API server"""
    uvicorn.run("api:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the Agent Village API server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    run_server(host=args.host, port=args.port, reload=args.reload)