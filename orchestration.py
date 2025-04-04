"""
Orchestration System Module - Coordination for the Agent Village

This module provides the high-level orchestration for running the Agent Village,
managing agent interactions, scheduling tasks, and monitoring the system.

Dependencies:
- Python 3.10+
- pydantic 2.3+
- asyncio
- aiohttp 3.8.4+
"""

import asyncio
import datetime
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable

import aiohttp
from pydantic import BaseModel, Field

from agent_framework import Agent, AgentConfig, AgentStatus, Message, MessageType, Tool, create_agent
from chatroom import Chatroom, ChatroomConfig, VirtualSpace
from memory_db import MemoryDatabase, MemoryConfig, MemoryType
from virtual_computer import VirtualComputer, VirtualComputerConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("orchestration.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class OrchestrationConfig(BaseModel):
    """Configuration for the Agent Village orchestration system"""
    village_name: str
    base_directory: str = "./agent_village"
    chatroom_config: Optional[ChatroomConfig] = None
    memory_config: Optional[MemoryConfig] = None
    max_concurrent_agents: int = 10
    agent_configs: List[Dict[str, Any]] = Field(default_factory=list)
    virtual_computers: List[Dict[str, Any]] = Field(default_factory=list)
    custom_configuration: Dict[str, Any] = Field(default_factory=dict)


class AgentVillage:
    """
    Orchestration system for managing the Agent Village
    
    This class provides high-level coordination for running the Agent Village,
    managing agent interactions, scheduling tasks, and monitoring the system.
    """
    def __init__(self, config: OrchestrationConfig):
        self.village_name = config.village_name
        self.base_directory = config.base_directory
        os.makedirs(self.base_directory, exist_ok=True)
        
        # Initialize chatroom
        chatroom_config = config.chatroom_config or ChatroomConfig(
            name=f"{self.village_name} Chatroom",
            description="Main communication channel for the Agent Village",
            persistence_path=os.path.join(self.base_directory, "chatroom_data")
        )
        self.chatroom = Chatroom(chatroom_config)
        
        # Initialize memory database
        memory_config = config.memory_config or MemoryConfig(
            storage_path=os.path.join(self.base_directory, "memory_data")
        )
        self.memory_db = MemoryDatabase(memory_config)
        
        # Initialize virtual computers
        self.virtual_computers = {}
        for computer_config in config.virtual_computers:
            vc_config = VirtualComputerConfig(**computer_config)
            computer = VirtualComputer(vc_config)
            self.virtual_computers[computer.name] = computer
            
            # Add computer as a resource to the chatroom
            self.chatroom.add_resource(computer.name, computer)
        
        # Check if we need to create a default computer
        if not self.virtual_computers:
            default_computer_config = VirtualComputerConfig(
                name="MainComputer",
                working_directory=os.path.join(self.base_directory, "computer_data")
            )
            default_computer = VirtualComputer(default_computer_config)
            self.virtual_computers[default_computer.name] = default_computer
            self.chatroom.add_resource("MainComputer", default_computer)
        
        # Create virtual space
        self.virtual_space = VirtualSpace(
            name=self.village_name,
            description=f"A collaborative environment for AI agents in {self.village_name}",
            chatroom=self.chatroom
        )
        
        # Agent tracking
        self.agents = {}
        self.agent_tasks = {}
        self.max_concurrent_agents = config.max_concurrent_agents
        
        # Project tracking
        self.projects = {}
        
        # System status
        self.status = "initializing"
        self.start_time = time.time()
        self.stats = {
            "messages_processed": 0,
            "tasks_completed": 0,
            "errors": 0
        }
        
        # Custom configuration
        self.custom_config = config.custom_configuration
        
        # Register callback for message processing
        self.chatroom.add_message_callback(self._on_message)
        
        # Initialize with provided agent configurations
        if config.agent_configs:
            for agent_config_dict in config.agent_configs:
                agent_config = AgentConfig(**agent_config_dict)
                self._create_agent(agent_config)
        
        logger.info(f"Initialized Agent Village: {self.village_name}")
    
    def _create_agent(self, config: AgentConfig) -> Optional[Agent]:
        """
        Create and initialize an agent
        
        Args:
            config: Configuration for the agent
            
        Returns:
            The created agent, or None if creation failed
        """
        try:
            # Check if name is already taken
            if config.name in self.agents:
                logger.warning(f"Agent name {config.name} is already taken")
                return None
            
            # Check if we've reached the maximum number of agents
            if len(self.agents) >= self.max_concurrent_agents:
                logger.warning(f"Maximum number of agents ({self.max_concurrent_agents}) reached")
                return None
            
            # Initialize LLM (this would be replaced with actual LLM initialization)
            # For now, we'll use a placeholder
            from langchain.llms import BaseLLM
            llm = BaseLLM()  # This is just a placeholder
            
            # Create the agent
            agent = create_agent(config, llm)
            
            # Add tools from virtual computers
            for computer_name, computer in self.virtual_computers.items():
                for tool_name, tool in computer.tools.items():
                    # Check if the agent has permission to use this tool
                    if tool_name in config.tools:
                        agent.add_tool(tool)
            
            # Add the agent to the chatroom
            self.chatroom.add_agent(agent)
            
            # Add to our tracking
            self.agents[agent.name] = agent
            
            # Add to virtual space (default to main room)
            self.virtual_space.move_agent_to_room(agent.name, "main")
            
            logger.info(f"Created agent {agent.name} with role {agent.role}")
            return agent
        except Exception as e:
            logger.error(f"Error creating agent: {str(e)}")
            return None
    
    def _on_message(self, message: Message) -> None:
        """
        Callback for when a message is sent in the chatroom
        
        Args:
            message: The message that was sent
        """
        # Update stats
        self.stats["messages_processed"] += 1
        
        # Store in memory database
        content = message.content.get("text", "") if isinstance(message.content, dict) else str(message.content)
        
        # Determine memory type
        if message.message_type == MessageType.CHAT:
            memory_type = MemoryType.CONVERSATION
        elif message.message_type == MessageType.THOUGHT:
            memory_type = MemoryType.THOUGHT
        elif message.message_type == MessageType.ACTION:
            memory_type = MemoryType.ACTION
        elif message.message_type == MessageType.SYSTEM:
            memory_type = MemoryType.OBSERVATION
        else:
            memory_type = MemoryType.OBSERVATION
        
        # Create metadata
        metadata = {
            "sender": message.sender,
            "recipient": message.recipient,
            "message_type": message.message_type.value,
            "timestamp": message.timestamp,
            "thread_id": message.thread_id,
            "in_reply_to": message.in_reply_to
        }
        
        # Add additional metadata from the message
        if isinstance(message.content, dict) and "metadata" in message.content:
            metadata.update(message.content["metadata"])
        
        # Store in memory database
        self.memory_db.add_memory(
            agent_id=message.sender,
            memory_type=memory_type,
            content=content,
            metadata=metadata,
            tags=["message", message.message_type.value]
        )
        
        logger.debug(f"Processed message from {message.sender} to {message.recipient or 'all'}")
    
    async def _initialize_project_manager(self) -> None:
        """Initialize the project manager agent for coordinating tasks"""
        # Create project manager config
        config = AgentConfig(
            name="ProjectManager",
            role="Project Manager",
            description="Coordinates tasks, assigns work, and ensures project goals are met",
            llm_model="gpt-4",
            llm_temperature=0.3,
            memory_path=os.path.join(self.base_directory, "memory_data"),
            tools=["list_files", "read_file", "create_project"],
            permissions=["read", "write", "assign_tasks"],
            system_prompt=(
                "You are the Project Manager for the Agent Village. Your role is to coordinate tasks, "
                "assign work to agents based on their skills, and ensure that project goals are met. "
                "You should maintain a high-level view of all activities and help agents collaborate effectively."
            )
        )
        
        # Create the project manager agent
        project_manager = self._create_agent(config)
        
        if project_manager:
            # Send a welcome message to all agents
            welcome_message = Message(
                sender="ProjectManager",
                recipient=None,  # Broadcast
                message_type=MessageType.SYSTEM,
                content={
                    "text": (
                        f"Hello team! I'm the Project Manager for {self.village_name}. "
                        f"I'll be coordinating our work and helping us achieve our goals. "
                        f"Let's collaborate effectively and build something amazing together!"
                    )
                }
            )
            
            await self.chatroom.process_message(welcome_message)
            logger.info("Project Manager initialized and sent welcome message")
        else:
            logger.error("Failed to initialize Project Manager")
    
    async def start(self) -> None:
        """Start the Agent Village system"""
        try:
            logger.info(f"Starting Agent Village: {self.village_name}")
            
            # Initialize database connections
            await self.chatroom.initialize_db()
            
            # Initialize project manager
            await self._initialize_project_manager()
            
            # Send system startup message
            startup_message = Message(
                sender="System",
                recipient=None,  # Broadcast
                message_type=MessageType.SYSTEM,
                content={
                    "text": f"Agent Village {self.village_name} is now starting. All systems initializing.",
                    "village_name": self.village_name,
                    "start_time": time.time(),
                    "agents": list(self.agents.keys())
                }
            )
            
            await self.chatroom.process_message(startup_message)
            
            # Start all agents
            self.status = "running"
            await self.chatroom.start_all_agents()
            
            logger.info(f"Agent Village {self.village_name} successfully started")
        except Exception as e:
            self.status = "error"
            logger.error(f"Error starting Agent Village: {str(e)}")
            self.stats["errors"] += 1
    
    async def stop(self) -> None:
        """Stop the Agent Village system"""
        try:
            logger.info(f"Stopping Agent Village: {self.village_name}")
            
            # Send system shutdown message
            shutdown_message = Message(
                sender="System",
                recipient=None,  # Broadcast
                message_type=MessageType.SYSTEM,
                content={
                    "text": f"Agent Village {self.village_name} is shutting down. Thank you for your collaboration.",
                    "village_name": self.village_name,
                    "shutdown_time": time.time(),
                    "uptime_seconds": time.time() - self.start_time
                }
            )
            
            await self.chatroom.process_message(shutdown_message)
            
            # Stop all agents
            await self.chatroom.stop_all_agents()
            
            # Cleanup virtual computers
            for computer in self.virtual_computers.values():
                computer.cleanup()
            
            self.status = "stopped"
            logger.info(f"Agent Village {self.village_name} successfully stopped")
        except Exception as e:
            self.status = "error"
            logger.error(f"Error stopping Agent Village: {str(e)}")
            self.stats["errors"] += 1
    
    async def add_agent(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add a new agent to the village
        
        Args:
            config: Configuration for the agent
            
        Returns:
            Dictionary containing result of operation
        """
        try:
            # Create agent config
            agent_config = AgentConfig(**config)
            
            # Create the agent
            agent = self._create_agent(agent_config)
            
            if agent:
                # Send system message about new agent
                new_agent_message = Message(
                    sender="System",
                    recipient=None,  # Broadcast
                    message_type=MessageType.SYSTEM,
                    content={
                        "text": f"A new agent has joined the village: {agent.name} ({agent.role})",
                        "agent_name": agent.name,
                        "agent_role": agent.role
                    }
                )
                
                await self.chatroom.process_message(new_agent_message)
                
                return {
                    "success": True,
                    "agent_name": agent.name,
                    "agent_id": agent.id,
                    "agent_role": agent.role
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to create agent"
                }
        except Exception as e:
            logger.error(f"Error adding agent: {str(e)}")
            self.stats["errors"] += 1
            return {
                "success": False,
                "error": str(e)
            }
    
    async def remove_agent(self, agent_name: str) -> Dict[str, Any]:
        """
        Remove an agent from the village
        
        Args:
            agent_name: Name of the agent to remove
            
        Returns:
            Dictionary containing result of operation
        """
        try:
            if agent_name not in self.agents:
                return {
                    "success": False,
                    "error": f"Agent {agent_name} not found"
                }
            
            # Get agent
            agent = self.agents[agent_name]
            
            # Send farewell message
            farewell_message = Message(
                sender="System",
                recipient=None,  # Broadcast
                message_type=MessageType.SYSTEM,
                content={
                    "text": f"Agent {agent_name} ({agent.role}) is leaving the village.",
                    "agent_name": agent_name,
                    "agent_role": agent.role
                }
            )
            
            await self.chatroom.process_message(farewell_message)
            
            # Remove from chatroom
            self.chatroom.remove_agent(agent_name)
            
            # Remove from tracking
            del self.agents[agent_name]
            
            return {
                "success": True,
                "agent_name": agent_name
            }
        except Exception as e:
            logger.error(f"Error removing agent {agent_name}: {str(e)}")
            self.stats["errors"] += 1
            return {
                "success": False,
                "error": str(e)
            }
    
    async def create_project(self, name: str, description: str, goals: List[str],
                          team_members: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Create a new project in the village
        
        Args:
            name: Name of the project
            description: Description of the project
            goals: List of project goals
            team_members: List of agent names to assign to the project
            
        Returns:
            Dictionary containing result of operation
        """
        try:
            # Generate project ID
            project_id = str(uuid.uuid4())
            
            # Validate team members
            valid_team_members = []
            invalid_team_members = []
            
            if team_members:
                for member in team_members:
                    if member in self.agents:
                        valid_team_members.append(member)
                    else:
                        invalid_team_members.append(member)
            
            # Create project in virtual space
            self.virtual_space.create_project(
                project_id=project_id,
                name=name,
                description=description,
                manager="ProjectManager",
                team_members=valid_team_members
            )
            
            # Create project tracking
            self.projects[project_id] = {
                "id": project_id,
                "name": name,
                "description": description,
                "goals": goals,
                "team_members": valid_team_members,
                "created_at": time.time(),
                "status": "active"
            }
            
            # Send project creation message
            project_message = Message(
                sender="System",
                recipient=None,  # Broadcast
                message_type=MessageType.SYSTEM,
                content={
                    "text": f"New project created: {name}\nDescription: {description}\nGoals: {', '.join(goals)}",
                    "project_id": project_id,
                    "project_name": name,
                    "project_description": description,
                    "project_goals": goals,
                    "team_members": valid_team_members
                }
            )
            
            await self.chatroom.process_message(project_message)
            
            # Notify project manager
            pm_message = Message(
                sender="System",
                recipient="ProjectManager",
                message_type=MessageType.COMMAND,
                content={
                    "text": f"Please initialize project: {name}",
                    "command": "initialize_project",
                    "project_id": project_id,
                    "project_name": name,
                    "project_description": description,
                    "project_goals": goals,
                    "team_members": valid_team_members
                }
            )
            
            await self.chatroom.process_message(pm_message)
            
            return {
                "success": True,
                "project_id": project_id,
                "project_name": name,
                "team_members": valid_team_members,
                "invalid_team_members": invalid_team_members
            }
        except Exception as e:
            logger.error(f"Error creating project: {str(e)}")
            self.stats["errors"] += 1
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the Agent Village
        
        Returns:
            Dictionary containing status information
        """
        # Calculate uptime
        uptime_seconds = time.time() - self.start_time
        hours, remainder = divmod(uptime_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Get agent statistics
        agent_stats = {}
        for agent_name, agent in self.agents.items():
            agent_stats[agent_name] = {
                "role": agent.role,
                "status": agent.status.value if hasattr(agent.status, "value") else str(agent.status),
                "metrics": agent.get_metrics()
            }
        
        # Get project statistics
        project_stats = {}
        for project_id, project in self.projects.items():
            project_stats[project_id] = {
                "name": project["name"],
                "status": project["status"],
                "team_size": len(project["team_members"]),
                "goals": len(project["goals"]),
                "created_at": project["created_at"]
            }
        
        # Compile status information
        status_info = {
            "village_name": self.village_name,
            "status": self.status,
            "start_time": self.start_time,
            "current_time": time.time(),
            "uptime": {
                "seconds": uptime_seconds,
                "formatted": f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
            },
            "agents": {
                "total": len(self.agents),
                "max": self.max_concurrent_agents,
                "stats": agent_stats
            },
            "projects": {
                "total": len(self.projects),
                "stats": project_stats
            },
            "messages": self.stats["messages_processed"],
            "tasks_completed": self.stats["tasks_completed"],
            "errors": self.stats["errors"]
        }
        
        return status_info
    
    async def send_message(self, from_name: str, to_name: Optional[str], 
                         content: str, message_type: str = "chat") -> Dict[str, Any]:
        """
        Send a message in the Agent Village
        
        Args:
            from_name: Name of the sender
            to_name: Name of the recipient (None for broadcast)
            content: Message content
            message_type: Type of message
            
        Returns:
            Dictionary containing result of operation
        """
        try:
            # Validate sender
            if from_name != "System" and from_name not in self.agents:
                return {
                    "success": False,
                    "error": f"Sender {from_name} not found"
                }
            
            # Validate recipient if not broadcast
            if to_name and to_name not in self.agents:
                return {
                    "success": False,
                    "error": f"Recipient {to_name} not found"
                }
            
            # Determine message type
            try:
                msg_type = MessageType(message_type)
            except ValueError:
                msg_type = MessageType.CHAT
            
            # Create message
            message = Message(
                sender=from_name,
                recipient=to_name,
                message_type=msg_type,
                content={
                    "text": content
                }
            )
            
            # Process message
            await self.chatroom.process_message(message)
            
            return {
                "success": True,
                "message_id": message.id,
                "sender": from_name,
                "recipient": to_name,
                "timestamp": message.timestamp
            }
        except Exception as e:
            logger.error(f"Error sending message: {str(e)}")
            self.stats["errors"] += 1
            return {
                "success": False,
                "error": str(e)
            }
    
    async def execute_computer_command(self, computer_name: str, tool_name: str,
                                    args: List[Any], kwargs: Dict[str, Any],
                                    agent_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute a command on a virtual computer
        
        Args:
            computer_name: Name of the computer
            tool_name: Name of the tool to use
            args: Positional arguments for the tool
            kwargs: Keyword arguments for the tool
            agent_name: Name of the agent executing the command (for permission checks)
            
        Returns:
            Dictionary containing result of operation
        """
        try:
            # Validate computer
            if computer_name not in self.virtual_computers:
                return {
                    "success": False,
                    "error": f"Computer {computer_name} not found"
                }
            
            computer = self.virtual_computers[computer_name]
            
            # Validate tool
            tool = computer.get_tool(tool_name)
            if not tool:
                return {
                    "success": False,
                    "error": f"Tool {tool_name} not found on computer {computer_name}"
                }
            
            # Check permissions if agent is specified
            if agent_name and agent_name != "System":
                if agent_name not in self.agents:
                    return {
                        "success": False,
                        "error": f"Agent {agent_name} not found"
                    }
                
                agent = self.agents[agent_name]
                
                # Check if agent has the tool
                has_tool = False
                for agent_tool in agent.tools.values():
                    if agent_tool.name == tool_name:
                        has_tool = True
                        break
                
                if not has_tool:
                    return {
                        "success": False,
                        "error": f"Agent {agent_name} does not have permission to use tool {tool_name}"
                    }
            
            # Execute the tool
            result = tool.execute(*args, **kwargs)
            
            # Log execution
            if agent_name:
                logger.info(f"Agent {agent_name} executed {tool_name} on {computer_name}")
                
                # Store in memory if agent is specified
                if agent_name in self.agents:
                    # Convert result to string for storage
                    result_str = json.dumps(result, indent=2)
                    
                    self.memory_db.add_memory(
                        agent_id=agent_name,
                        memory_type=MemoryType.ACTION,
                        content=f"Executed {tool_name} on {computer_name} with result: {result_str}",
                        metadata={
                            "computer": computer_name,
                            "tool": tool_name,
                            "args": args,
                            "kwargs": kwargs,
                            "success": "success" in result and result["success"]
                        },
                        tags=["command", "tool", tool_name, computer_name]
                    )
            
            return {
                "success": True,
                "computer": computer_name,
                "tool": tool_name,
                "result": result
            }
        except Exception as e:
            logger.error(f"Error executing command on {computer_name}: {str(e)}")
            self.stats["errors"] += 1
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_recent_messages(self, limit: int = 50, 
                          agent_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get recent messages from the chatroom
        
        Args:
            limit: Maximum number of messages to retrieve
            agent_name: Filter to messages involving this agent
            
        Returns:
            Dictionary containing messages
        """
        try:
            # Get messages asynchronously but run in a blocking manner
            loop = asyncio.get_event_loop()
            messages = loop.run_until_complete(
                self.chatroom.get_message_history(
                    limit=limit,
                    sender=agent_name,
                    recipient=agent_name
                )
            )
            
            # Convert messages to dictionaries
            message_dicts = [message.to_dict() for message in messages]
            
            return {
                "success": True,
                "messages": message_dicts,
                "count": len(message_dicts),
                "limit": limit
            }
        except Exception as e:
            logger.error(f"Error retrieving recent messages: {str(e)}")
            self.stats["errors"] += 1
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_agent_memories(self, agent_name: str, memory_type: Optional[str] = None,
                         limit: int = 20) -> Dict[str, Any]:
        """
        Get memories for an agent
        
        Args:
            agent_name: Name of the agent
            memory_type: Type of memories to retrieve
            limit: Maximum number of memories to retrieve
            
        Returns:
            Dictionary containing memories
        """
        try:
            # Validate agent
            if agent_name not in self.agents:
                return {
                    "success": False,
                    "error": f"Agent {agent_name} not found"
                }
            
            # Convert memory type string to enum if provided
            memory_type_enum = None
            if memory_type:
                try:
                    memory_type_enum = MemoryType(memory_type)
                except ValueError:
                    return {
                        "success": False,
                        "error": f"Invalid memory type: {memory_type}"
                    }
            
            # Get memories
            memories = self.memory_db.get_agent_memories(
                agent_id=agent_name,
                memory_type=memory_type_enum,
                limit=limit
            )
            
            return {
                "success": True,
                "agent_name": agent_name,
                "memories": memories,
                "count": len(memories),
                "limit": limit
            }
        except Exception as e:
            logger.error(f"Error retrieving memories for agent {agent_name}: {str(e)}")
            self.stats["errors"] += 1
            return {
                "success": False,
                "error": str(e)
            }
    
    def search_memories(self, query: str, agent_name: Optional[str] = None,
                      limit: int = 10) -> Dict[str, Any]:
        """
        Search agent memories
        
        Args:
            query: Search query
            agent_name: Filter to a specific agent
            limit: Maximum number of results
            
        Returns:
            Dictionary containing search results
        """
        try:
            # Perform search
            results = self.memory_db.search_by_content(
                query=query,
                agent_id=agent_name,
                n_results=limit
            )
            
            return {
                "success": True,
                "query": query,
                "agent_name": agent_name,
                "results": results,
                "count": len(results),
                "limit": limit
            }
        except Exception as e:
            logger.error(f"Error searching memories: {str(e)}")
            self.stats["errors"] += 1
            return {
                "success": False,
                "error": str(e)
            }