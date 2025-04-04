"""
Environment and Chatroom Module - Shared space for agent interaction

This module provides the communication environment for agents to interact with
each other and with the external world. It manages message passing, scheduling,
and coordination between agents.

Dependencies:
- Python 3.10+
- asyncio
- pydantic 2.3+
- SQLAlchemy 2.0+
"""

import asyncio
import datetime
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable

import aiosqlite
from pydantic import BaseModel, Field

from agent_framework import Agent, Message, MessageType, Tool


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("environment.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class ChatroomConfig(BaseModel):
    """Configuration for a chatroom/environment"""
    name: str
    description: str
    persistence_path: str = "./chatroom_storage"
    max_history: int = 1000
    broadcast_system_messages: bool = True
    log_messages: bool = True
    max_message_size: int = 100000  # bytes
    custom_configuration: Dict[str, Any] = Field(default_factory=dict)


class Chatroom:
    """
    A shared communication environment for agents
    
    This class manages communication between agents, maintains message history,
    and provides facilities for persisting conversations.
    """
    def __init__(self, config: ChatroomConfig):
        self.id = str(uuid.uuid4())
        self.name = config.name
        self.description = config.description
        self.persistence_path = os.path.join(config.persistence_path, f"chatroom_{self.id}")
        os.makedirs(self.persistence_path, exist_ok=True)
        
        self.messages: List[Message] = []
        self.agents: Dict[str, Agent] = {}
        self.resources: Dict[str, Any] = {}
        
        self.max_history = config.max_history
        self.broadcast_system_messages = config.broadcast_system_messages
        self.log_messages = config.log_messages
        self.max_message_size = config.max_message_size
        
        self.custom_config = config.custom_configuration
        
        # Event handling and callbacks
        self.message_callbacks: List[Callable[[Message], None]] = []
        self.agent_join_callbacks: List[Callable[[str], None]] = []
        self.agent_leave_callbacks: List[Callable[[str], None]] = []
        
        # Database connection for message persistence
        self.db_path = os.path.join(self.persistence_path, "messages.db")
        self.db_initialized = False
        
        logger.info(f"Initialized chatroom {self.name} ({self.id})")
    
    async def initialize_db(self) -> None:
        """Initialize the SQLite database for message persistence"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    timestamp REAL,
                    sender TEXT,
                    recipient TEXT,
                    message_type TEXT,
                    content TEXT,
                    thread_id TEXT,
                    in_reply_to TEXT,
                    metadata TEXT
                )
            ''')
            await db.commit()
        
        self.db_initialized = True
        logger.info(f"Initialized database for chatroom {self.name}")
    
    def add_agent(self, agent: Agent) -> None:
        """
        Add an agent to the chatroom
        
        Args:
            agent: The agent to add
        """
        if agent.name in self.agents:
            logger.warning(f"Agent {agent.name} already exists in chatroom {self.name}")
            return
        
        self.agents[agent.name] = agent
        logger.info(f"Added agent {agent.name} to chatroom {self.name}")
        
        # Notify callbacks
        for callback in self.agent_join_callbacks:
            try:
                callback(agent.name)
            except Exception as e:
                logger.error(f"Error in agent join callback: {str(e)}")
    
    def remove_agent(self, agent_name: str) -> None:
        """
        Remove an agent from the chatroom
        
        Args:
            agent_name: The name of the agent to remove
        """
        if agent_name not in self.agents:
            logger.warning(f"Agent {agent_name} does not exist in chatroom {self.name}")
            return
        
        del self.agents[agent_name]
        logger.info(f"Removed agent {agent_name} from chatroom {self.name}")
        
        # Notify callbacks
        for callback in self.agent_leave_callbacks:
            try:
                callback(agent_name)
            except Exception as e:
                logger.error(f"Error in agent leave callback: {str(e)}")
    
    def add_resource(self, resource_name: str, resource: Any) -> None:
        """
        Add a shared resource to the chatroom
        
        Args:
            resource_name: Name of the resource
            resource: The resource to add
        """
        self.resources[resource_name] = resource
        logger.info(f"Added resource {resource_name} to chatroom {self.name}")
    
    def get_resource(self, resource_name: str) -> Optional[Any]:
        """
        Get a shared resource from the chatroom
        
        Args:
            resource_name: Name of the resource
            
        Returns:
            The resource if it exists, None otherwise
        """
        return self.resources.get(resource_name)
    
    def add_message_callback(self, callback: Callable[[Message], None]) -> None:
        """
        Add a callback to be called when a message is received
        
        Args:
            callback: The callback function
        """
        self.message_callbacks.append(callback)
    
    def add_agent_join_callback(self, callback: Callable[[str], None]) -> None:
        """
        Add a callback to be called when an agent joins the chatroom
        
        Args:
            callback: The callback function
        """
        self.agent_join_callbacks.append(callback)
    
    def add_agent_leave_callback(self, callback: Callable[[str], None]) -> None:
        """
        Add a callback to be called when an agent leaves the chatroom
        
        Args:
            callback: The callback function
        """
        self.agent_leave_callbacks.append(callback)
    
    async def broadcast_message(self, message: Message) -> None:
        """
        Broadcast a message to all agents in the chatroom
        
        Args:
            message: The message to broadcast
        """
        # Set recipient to None for broadcast
        if message.recipient is not None:
            message = Message(
                id=message.id,
                timestamp=message.timestamp,
                sender=message.sender,
                recipient=None,  # Set to None for broadcast
                message_type=message.message_type,
                content=message.content,
                thread_id=message.thread_id,
                in_reply_to=message.in_reply_to,
                metadata=message.metadata
            )
        
        # Add to message history
        self.messages.append(message)
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]
        
        # Persist the message if logging is enabled
        if self.log_messages:
            await self._persist_message(message)
        
        # Notify callbacks
        for callback in self.message_callbacks:
            try:
                callback(message)
            except Exception as e:
                logger.error(f"Error in message callback: {str(e)}")
        
        # Distribute to all agents
        tasks = []
        for agent_name, agent in self.agents.items():
            tasks.append(agent.receive_message(message))
        
        if tasks:
            await asyncio.gather(*tasks)
        
        logger.debug(f"Broadcast message {message.id} from {message.sender} to all agents")
    
    async def direct_message(self, message: Message) -> None:
        """
        Send a message to a specific agent
        
        Args:
            message: The message to send
        """
        if message.recipient is None:
            await self.broadcast_message(message)
            return
        
        # Check if recipient exists
        if message.recipient not in self.agents:
            logger.warning(f"Recipient {message.recipient} not found in chatroom {self.name}")
            # Notify sender that recipient doesn't exist
            if message.sender in self.agents:
                error_message = Message(
                    sender="System",
                    recipient=message.sender,
                    message_type=MessageType.ERROR,
                    content={
                        "text": f"Recipient {message.recipient} not found in chatroom {self.name}",
                        "original_message_id": message.id
                    }
                )
                await self.agents[message.sender].receive_message(error_message)
            return
        
        # Add to message history
        self.messages.append(message)
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]
        
        # Persist the message if logging is enabled
        if self.log_messages:
            await self._persist_message(message)
        
        # Notify callbacks
        for callback in self.message_callbacks:
            try:
                callback(message)
            except Exception as e:
                logger.error(f"Error in message callback: {str(e)}")
        
        # Send to recipient
        await self.agents[message.recipient].receive_message(message)
        
        logger.debug(f"Sent message {message.id} from {message.sender} to {message.recipient}")
    
    async def system_message(self, content: Dict[str, Any], message_type: MessageType = MessageType.SYSTEM) -> None:
        """
        Send a system message to all agents
        
        Args:
            content: Content of the message
            message_type: Type of the message
        """
        message = Message(
            sender="System",
            recipient=None,
            message_type=message_type,
            content=content
        )
        
        await self.broadcast_message(message)
    
    async def process_message(self, message: Message) -> None:
        """
        Process an incoming message
        
        Args:
            message: The message to process
        """
        # Validate message size
        message_size = len(json.dumps(message.to_dict()))
        if message_size > self.max_message_size:
            logger.warning(f"Message {message.id} exceeds maximum size of {self.max_message_size} bytes")
            # Notify sender
            if message.sender in self.agents:
                error_message = Message(
                    sender="System",
                    recipient=message.sender,
                    message_type=MessageType.ERROR,
                    content={
                        "text": f"Message exceeds maximum size of {self.max_message_size} bytes",
                        "original_message_id": message.id
                    }
                )
                await self.agents[message.sender].receive_message(error_message)
            return
        
        # Check if it's a direct message or broadcast
        if message.recipient is not None:
            await self.direct_message(message)
        else:
            await self.broadcast_message(message)
    
    async def _persist_message(self, message: Message) -> None:
        """
        Persist a message to the database
        
        Args:
            message: The message to persist
        """
        if not self.db_initialized:
            await self.initialize_db()
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    '''
                    INSERT INTO messages 
                    (id, timestamp, sender, recipient, message_type, content, thread_id, in_reply_to, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''',
                    (
                        message.id,
                        message.timestamp,
                        message.sender,
                        message.recipient,
                        message.message_type.value,
                        json.dumps(message.content),
                        message.thread_id,
                        message.in_reply_to,
                        json.dumps(message.metadata)
                    )
                )
                await db.commit()
        except Exception as e:
            logger.error(f"Error persisting message {message.id}: {str(e)}")
    
    async def get_message_history(self, limit: int = 100, offset: int = 0, 
                                 sender: Optional[str] = None, 
                                 recipient: Optional[str] = None,
                                 message_type: Optional[MessageType] = None,
                                 thread_id: Optional[str] = None) -> List[Message]:
        """
        Get message history with filtering options
        
        Args:
            limit: Maximum number of messages to retrieve
            offset: Offset for pagination
            sender: Filter by sender
            recipient: Filter by recipient
            message_type: Filter by message type
            thread_id: Filter by thread ID
            
        Returns:
            List of messages matching the filters
        """
        if not self.db_initialized:
            await self.initialize_db()
        
        query = "SELECT * FROM messages WHERE 1=1"
        params = []
        
        if sender:
            query += " AND sender = ?"
            params.append(sender)
        
        if recipient:
            query += " AND (recipient = ? OR recipient IS NULL)"
            params.append(recipient)
        
        if message_type:
            query += " AND message_type = ?"
            params.append(message_type.value)
        
        if thread_id:
            query += " AND thread_id = ?"
            params.append(thread_id)
        
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        messages = []
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, params) as cursor:
                async for row in cursor:
                    message = Message(
                        id=row["id"],
                        timestamp=row["timestamp"],
                        sender=row["sender"],
                        recipient=row["recipient"],
                        message_type=MessageType(row["message_type"]),
                        content=json.loads(row["content"]),
                        thread_id=row["thread_id"],
                        in_reply_to=row["in_reply_to"],
                        metadata=json.loads(row["metadata"])
                    )
                    messages.append(message)
        
        return messages
    
    def get_agent_names(self) -> List[str]:
        """
        Get the names of all agents in the chatroom
        
        Returns:
            List of agent names
        """
        return list(self.agents.keys())
    
    def get_agent(self, name: str) -> Optional[Agent]:
        """
        Get an agent by name
        
        Args:
            name: The name of the agent
            
        Returns:
            The agent if it exists, None otherwise
        """
        return self.agents.get(name)
    
    async def start_all_agents(self) -> None:
        """Start all agents in the chatroom"""
        tasks = []
        for agent in self.agents.values():
            task = asyncio.create_task(agent.run())
            tasks.append(task)
        
        logger.info(f"Started all agents in chatroom {self.name}")
        
        # Create welcome message
        await self.system_message({
            "text": f"Welcome to chatroom {self.name}!",
            "description": self.description,
            "agents": self.get_agent_names()
        })
        
        # Wait for all agents to complete (this will run until an exception or cancellation)
        if tasks:
            await asyncio.gather(*tasks)
    
    async def stop_all_agents(self) -> None:
        """Stop all agents in the chatroom"""
        for agent in self.agents.values():
            agent.status = "terminated"
        
        logger.info(f"Stopped all agents in chatroom {self.name}")
        
        # Create goodbye message
        await self.system_message({
            "text": f"Chatroom {self.name} is shutting down.",
        })


class VirtualSpace:
    """
    Extended environment with spatial representation and collaboration context
    
    This class provides a more sophisticated environment model with spatial
    awareness, collaborative workspaces, and project-oriented features.
    """
    def __init__(
        self,
        name: str,
        description: str,
        chatroom: Chatroom,
    ):
        self.name = name
        self.description = description
        self.chatroom = chatroom
        
        # Spatial representation
        self.rooms: Dict[str, Dict[str, Any]] = {
            "main": {
                "name": "Main Hall",
                "description": "The central gathering space for all agents",
                "agents": set(),
                "resources": {}
            }
        }
        
        # Collaborative workspaces
        self.workspaces: Dict[str, Dict[str, Any]] = {}
        
        # Projects and tasks
        self.projects: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Initialized virtual space {self.name}")
    
    def create_room(self, room_id: str, name: str, description: str) -> None:
        """
        Create a new room in the virtual space
        
        Args:
            room_id: Unique ID for the room
            name: Name of the room
            description: Description of the room
        """
        if room_id in self.rooms:
            logger.warning(f"Room {room_id} already exists in virtual space {self.name}")
            return
        
        self.rooms[room_id] = {
            "name": name,
            "description": description,
            "agents": set(),
            "resources": {}
        }
        
        logger.info(f"Created room {name} ({room_id}) in virtual space {self.name}")
    
    def create_workspace(self, workspace_id: str, name: str, description: str, 
                         owner: Optional[str] = None,
                         collaborators: Optional[List[str]] = None) -> None:
        """
        Create a new collaborative workspace
        
        Args:
            workspace_id: Unique ID for the workspace
            name: Name of the workspace
            description: Description of the workspace
            owner: Name of the agent that owns the workspace
            collaborators: List of agent names that can collaborate in the workspace
        """
        if workspace_id in self.workspaces:
            logger.warning(f"Workspace {workspace_id} already exists in virtual space {self.name}")
            return
        
        self.workspaces[workspace_id] = {
            "name": name,
            "description": description,
            "owner": owner,
            "collaborators": set(collaborators or []),
            "resources": {},
            "documents": {}
        }
        
        logger.info(f"Created workspace {name} ({workspace_id}) in virtual space {self.name}")
    
    def create_project(self, project_id: str, name: str, description: str,
                      manager: Optional[str] = None,
                      team_members: Optional[List[str]] = None,
                      deadline: Optional[str] = None) -> None:
        """
        Create a new project
        
        Args:
            project_id: Unique ID for the project
            name: Name of the project
            description: Description of the project
            manager: Name of the agent managing the project
            team_members: List of agent names on the project team
            deadline: Optional deadline for the project
        """
        if project_id in self.projects:
            logger.warning(f"Project {project_id} already exists in virtual space {self.name}")
            return
        
        self.projects[project_id] = {
            "name": name,
            "description": description,
            "manager": manager,
            "team_members": set(team_members or []),
            "tasks": {},
            "resources": {},
            "deadline": deadline,
            "created_at": datetime.datetime.now().isoformat(),
            "status": "active"
        }
        
        logger.info(f"Created project {name} ({project_id}) in virtual space {self.name}")
    
    def add_task_to_project(self, project_id: str, task_id: str, 
                          description: str, assignee: Optional[str] = None,
                          deadline: Optional[str] = None) -> None:
        """
        Add a task to a project
        
        Args:
            project_id: ID of the project
            task_id: Unique ID for the task
            description: Description of the task
            assignee: Name of the agent assigned to the task
            deadline: Optional deadline for the task
        """
        if project_id not in self.projects:
            logger.warning(f"Project {project_id} does not exist in virtual space {self.name}")
            return
        
        project = self.projects[project_id]
        
        if task_id in project["tasks"]:
            logger.warning(f"Task {task_id} already exists in project {project_id}")
            return
        
        project["tasks"][task_id] = {
            "description": description,
            "assignee": assignee,
            "deadline": deadline,
            "created_at": datetime.datetime.now().isoformat(),
            "status": "pending",
            "updates": []
        }
        
        logger.info(f"Added task {task_id} to project {project_id} in virtual space {self.name}")
    
    async def move_agent_to_room(self, agent_name: str, room_id: str) -> None:
        """
        Move an agent to a different room
        
        Args:
            agent_name: Name of the agent to move
            room_id: ID of the destination room
        """
        if room_id not in self.rooms:
            logger.warning(f"Room {room_id} does not exist in virtual space {self.name}")
            return
        
        if agent_name not in self.chatroom.agents:
            logger.warning(f"Agent {agent_name} does not exist in chatroom {self.chatroom.name}")
            return
        
        # Remove from current room
        for room in self.rooms.values():
            if agent_name in room["agents"]:
                room["agents"].remove(agent_name)
        
        # Add to new room
        self.rooms[room_id]["agents"].add(agent_name)
        
        # Notify the agent and others in the room
        await self.chatroom.system_message({
            "text": f"{agent_name} has moved to {self.rooms[room_id]['name']}",
            "agent": agent_name,
            "room": self.rooms[room_id]["name"],
            "room_id": room_id
        })
        
        logger.info(f"Moved agent {agent_name} to room {room_id} in virtual space {self.name}")
    
    def get_agents_in_room(self, room_id: str) -> List[str]:
        """
        Get the names of all agents in a room
        
        Args:
            room_id: ID of the room
            
        Returns:
            List of agent names in the room
        """
        if room_id not in self.rooms:
            logger.warning(f"Room {room_id} does not exist in virtual space {self.name}")
            return []
        
        return list(self.rooms[room_id]["agents"])
    
    def get_agent_location(self, agent_name: str) -> Optional[str]:
        """
        Get the room where an agent is located
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            ID of the room where the agent is located, or None if not found
        """
        for room_id, room in self.rooms.items():
            if agent_name in room["agents"]:
                return room_id
        
        return None
    
    def add_document_to_workspace(self, workspace_id: str, document_id: str,
                                title: str, content: str,
                                author: Optional[str] = None) -> None:
        """
        Add a document to a workspace
        
        Args:
            workspace_id: ID of the workspace
            document_id: Unique ID for the document
            title: Title of the document
            content: Content of the document
            author: Name of the agent that authored the document
        """
        if workspace_id not in self.workspaces:
            logger.warning(f"Workspace {workspace_id} does not exist in virtual space {self.name}")
            return
        
        workspace = self.workspaces[workspace_id]
        
        if document_id in workspace["documents"]:
            logger.warning(f"Document {document_id} already exists in workspace {workspace_id}")
            return
        
        workspace["documents"][document_id] = {
            "title": title,
            "content": content,
            "author": author,
            "created_at": datetime.datetime.now().isoformat(),
            "updated_at": datetime.datetime.now().isoformat(),
            "version": 1,
            "history": []
        }
        
        logger.info(f"Added document {title} ({document_id}) to workspace {workspace_id} in virtual space {self.name}")
    
    def update_document(self, workspace_id: str, document_id: str,
                      content: str, editor: Optional[str] = None) -> None:
        """
        Update the content of a document
        
        Args:
            workspace_id: ID of the workspace
            document_id: ID of the document
            content: New content for the document
            editor: Name of the agent updating the document
        """
        if workspace_id not in self.workspaces:
            logger.warning(f"Workspace {workspace_id} does not exist in virtual space {self.name}")
            return
        
        workspace = self.workspaces[workspace_id]
        
        if document_id not in workspace["documents"]:
            logger.warning(f"Document {document_id} does not exist in workspace {workspace_id}")
            return
        
        document = workspace["documents"][document_id]
        
        # Save current version to history
        document["history"].append({
            "content": document["content"],
            "updated_at": document["updated_at"],
            "version": document["version"],
            "editor": document.get("editor")
        })
        
        # Update document
        document["content"] = content
        document["updated_at"] = datetime.datetime.now().isoformat()
        document["version"] += 1
        document["editor"] = editor
        
        logger.info(f"Updated document {document_id} in workspace {workspace_id} to version {document['version']}")
    
    def get_document(self, workspace_id: str, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a document from a workspace
        
        Args:
            workspace_id: ID of the workspace
            document_id: ID of the document
            
        Returns:
            The document if it exists, None otherwise
        """
        if workspace_id not in self.workspaces:
            logger.warning(f"Workspace {workspace_id} does not exist in virtual space {self.name}")
            return None
        
        workspace = self.workspaces[workspace_id]
        
        if document_id not in workspace["documents"]:
            logger.warning(f"Document {document_id} does not exist in workspace {workspace_id}")
            return None
        
        return workspace["documents"][document_id]
    
    def get_workspace_collaborators(self, workspace_id: str) -> List[str]:
        """
        Get the names of all collaborators in a workspace
        
        Args:
            workspace_id: ID of the workspace
            
        Returns:
            List of agent names that are collaborators in the workspace
        """
        if workspace_id not in self.workspaces:
            logger.warning(f"Workspace {workspace_id} does not exist in virtual space {self.name}")
            return []
        
        workspace = self.workspaces[workspace_id]
        collaborators = list(workspace["collaborators"])
        
        if workspace["owner"] and workspace["owner"] not in collaborators:
            collaborators.append(workspace["owner"])
        
        return collaborators