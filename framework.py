"""
Agent Framework Module - Foundation for all intelligent agents in the system

This module provides the base classes and implementations for creating specialized
intelligent agents within the Agent Village ecosystem. It handles agent lifecycle,
communication, perception, cognition, and action execution.

Dependencies:
- Python 3.10+
- PyTorch 2.0+
- langchain 0.0.267+
- pydantic 2.3+
- SQLAlchemy 2.0+
"""

import asyncio
import datetime
import json
import logging
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import torch
from langchain.llms import BaseLLM
from langchain.memory import VectorStoreRetrieverMemory
from langchain.callbacks.manager import CallbackManager
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field, validator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("agent_village.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Represents the current operational status of an agent"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PAUSED = "paused"
    TERMINATED = "terminated"
    ERROR = "error"


class MessageType(Enum):
    """Defines the types of messages agents can exchange"""
    CHAT = "chat"            # Regular conversation
    COMMAND = "command"      # Command to be executed
    RESULT = "result"        # Result of an executed command
    SYSTEM = "system"        # System message
    STATUS = "status"        # Status update
    QUERY = "query"          # Information request
    RESPONSE = "response"    # Response to a query
    ERROR = "error"          # Error message


class Message(BaseModel):
    """Standardized message format for agent communication"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = Field(default_factory=lambda: datetime.datetime.now().timestamp())
    sender: str
    recipient: Optional[str] = None  # None means broadcast
    message_type: MessageType
    content: Dict[str, Any]
    thread_id: Optional[str] = None  # For tracking conversation threads
    in_reply_to: Optional[str] = None  # ID of the message this is replying to
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format"""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "sender": self.sender,
            "recipient": self.recipient,
            "message_type": self.message_type.value,
            "content": self.content,
            "thread_id": self.thread_id,
            "in_reply_to": self.in_reply_to,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create a Message from dictionary data"""
        # Convert string message_type back to enum
        if isinstance(data["message_type"], str):
            data["message_type"] = MessageType(data["message_type"])
        return cls(**data)


class AgentMemory:
    """
    Memory system for agents to store and retrieve information
    Handles encoding, storage, and retrieval of agent experiences
    """
    def __init__(
        self,
        agent_id: str,
        storage_path: str = "./memory_storage",
        embedding_model=None,
        max_tokens_limit: int = 4000,
    ):
        self.agent_id = agent_id
        self.storage_path = os.path.join(storage_path, f"agent_{agent_id}")
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Initialize embedding model
        self.embedding_model = embedding_model or OpenAIEmbeddings()
        
        # Initialize vector database
        self.vectorstore = Chroma(
            collection_name=f"agent_{agent_id}_memory",
            embedding_function=self.embedding_model,
            persist_directory=self.storage_path
        )
        
        # Initialize retriever
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 5}
        )
        
        # Initialize memory
        self.memory = VectorStoreRetrieverMemory(
            retriever=self.retriever,
        )
        
        # Short-term memory (recent interactions)
        self.short_term_memory: List[Message] = []
        self.max_short_term_items = 50
        
        # Token tracking
        self.max_tokens_limit = max_tokens_limit
        self.token_count = 0
        
        # Working memory (current task state)
        self.working_memory: Dict[str, Any] = {}
        
        logger.info(f"Initialized memory system for agent {agent_id}")
    
    def add(self, message: Message) -> None:
        """
        Add a message to both short-term and long-term memory
        
        Args:
            message: The message to store in memory
        """
        # Add to short term memory
        self.short_term_memory.append(message)
        
        # Trim short term memory if needed
        if len(self.short_term_memory) > self.max_short_term_items:
            self.short_term_memory = self.short_term_memory[-self.max_short_term_items:]
        
        # Add to vector store for long-term memory
        text_representation = f"Time: {datetime.datetime.fromtimestamp(message.timestamp).isoformat()}\n"
        text_representation += f"From: {message.sender}\n"
        text_representation += f"To: {message.recipient or 'All'}\n"
        text_representation += f"Type: {message.message_type.value}\n"
        text_representation += f"Content: {json.dumps(message.content)}"
        
        self.memory.save_context(
            {"input": f"Message received at {datetime.datetime.fromtimestamp(message.timestamp).isoformat()}"},
            {"output": text_representation}
        )
        
        logger.debug(f"Agent {self.agent_id} stored message {message.id} in memory")
    
    def get_relevant_context(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context based on a query
        
        Args:
            query: The query to search for relevant context
            k: Number of relevant items to retrieve
            
        Returns:
            List of relevant memory items
        """
        docs = self.vectorstore.similarity_search(query, k=k)
        return [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
    
    def get_recent_messages(self, n: int = 10) -> List[Message]:
        """
        Get the n most recent messages from short-term memory
        
        Args:
            n: Number of recent messages to retrieve
            
        Returns:
            List of the n most recent messages
        """
        return self.short_term_memory[-n:]
    
    def set_working_memory(self, key: str, value: Any) -> None:
        """
        Store a value in working memory
        
        Args:
            key: The key to store the value under
            value: The value to store
        """
        self.working_memory[key] = value
    
    def get_working_memory(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a value from working memory
        
        Args:
            key: The key to retrieve
            default: Default value if key doesn't exist
            
        Returns:
            The stored value or default
        """
        return self.working_memory.get(key, default)
    
    def clear_working_memory(self) -> None:
        """Clear all items in working memory"""
        self.working_memory = {}
    
    def save(self) -> None:
        """Persist memory to disk"""
        self.vectorstore.persist()
        
        # Save short-term memory
        with open(os.path.join(self.storage_path, "short_term.json"), "w") as f:
            json.dump([m.to_dict() for m in self.short_term_memory], f)
        
        # Save working memory
        with open(os.path.join(self.storage_path, "working_memory.json"), "w") as f:
            json.dump(self.working_memory, f)
            
        logger.info(f"Agent {self.agent_id} memory saved to disk")
    
    def load(self) -> None:
        """Load memory from disk"""
        # Load short-term memory
        short_term_path = os.path.join(self.storage_path, "short_term.json")
        if os.path.exists(short_term_path):
            with open(short_term_path, "r") as f:
                data = json.load(f)
                self.short_term_memory = [Message.from_dict(m) for m in data]
        
        # Load working memory
        working_memory_path = os.path.join(self.storage_path, "working_memory.json")
        if os.path.exists(working_memory_path):
            with open(working_memory_path, "r") as f:
                self.working_memory = json.load(f)
                
        logger.info(f"Agent {self.agent_id} memory loaded from disk")


class Tool(BaseModel):
    """Tool/capability that an agent can use to interact with environment"""
    name: str
    description: str
    function: Callable
    parameters: Dict[str, Any] = Field(default_factory=dict)
    required_permissions: List[str] = Field(default_factory=list)
    
    def execute(self, *args, **kwargs) -> Any:
        """Execute the tool with the given arguments"""
        try:
            logger.debug(f"Executing tool {self.name} with args: {args}, kwargs: {kwargs}")
            result = self.function(*args, **kwargs)
            return result
        except Exception as e:
            logger.error(f"Error executing tool {self.name}: {str(e)}")
            raise


class AgentConfig(BaseModel):
    """Configuration settings for an agent"""
    name: str
    role: str
    description: str
    llm_model: str
    llm_temperature: float = 0.7
    llm_max_tokens: int = 1024
    memory_path: str = "./memory_storage"
    tools: List[str] = Field(default_factory=list)
    permissions: List[str] = Field(default_factory=list)
    system_prompt: str = ""
    custom_configuration: Dict[str, Any] = Field(default_factory=dict)


class Agent(ABC):
    """
    Base class for all agents in the system
    
    This abstract class defines the interface and basic functionality
    for all intelligent agents in the Agent Village ecosystem.
    """
    def __init__(
        self,
        config: AgentConfig,
        llm: BaseLLM,
        callback_manager: Optional[CallbackManager] = None,
    ):
        self.id = str(uuid.uuid4())
        self.name = config.name
        self.role = config.role
        self.description = config.description
        self.status = AgentStatus.INITIALIZING
        
        # Set up LLM
        self.llm = llm
        self.llm_temperature = config.llm_temperature
        self.llm_max_tokens = config.llm_max_tokens
        
        # Set up memory
        self.memory = AgentMemory(self.id, config.memory_path)
        
        # Set up tools and permissions
        self.tools: Dict[str, Tool] = {}
        self.permissions = set(config.permissions)
        
        # Communication channels
        self.message_queue = asyncio.Queue()
        self.response_callbacks: Dict[str, Callable] = {}
        
        # System prompt
        self.system_prompt = config.system_prompt
        if not self.system_prompt:
            self.system_prompt = self._generate_default_system_prompt()
            
        # Custom configuration
        self.custom_config = config.custom_configuration
        
        # Set up callback manager
        self.callback_manager = callback_manager
        
        # Performance metrics
        self.metrics = {
            "messages_received": 0,
            "messages_sent": 0,
            "tools_used": 0,
            "errors_encountered": 0,
            "tasks_completed": 0,
            "response_time": []
        }
        
        logger.info(f"Initialized agent {self.name} ({self.id}) with role {self.role}")
        
    def _generate_default_system_prompt(self) -> str:
        """Generate a default system prompt based on agent information"""
        prompt = f"You are {self.name}, a {self.role} in the Agent Village system. "
        prompt += f"\n\nDescription of your role: {self.description}"
        prompt += "\n\nYou collaborate with other AI agents to achieve collective goals. "
        prompt += "You should always act in accordance with your role and responsibilities."
        prompt += "\n\nWhen communicating with other agents, be clear, concise, and helpful."
        return prompt
    
    def add_tool(self, tool: Tool) -> None:
        """
        Add a tool that the agent can use
        
        Args:
            tool: The tool to add
        """
        # Check if agent has permissions to use this tool
        for permission in tool.required_permissions:
            if permission not in self.permissions:
                logger.warning(f"Agent {self.name} does not have permission {permission} required for tool {tool.name}")
                return
        
        self.tools[tool.name] = tool
        logger.info(f"Added tool {tool.name} to agent {self.name}")
    
    async def receive_message(self, message: Message) -> None:
        """
        Receive a message and add it to the message queue
        
        Args:
            message: The message to receive
        """
        # Validate message is for this agent
        if message.recipient and message.recipient != self.name and message.recipient != "all":
            logger.warning(f"Agent {self.name} received message intended for {message.recipient}")
            return
        
        # Log message receipt
        logger.debug(f"Agent {self.name} received message: {message.id} from {message.sender}")
        
        # Store in memory
        self.memory.add(message)
        
        # Update metrics
        self.metrics["messages_received"] += 1
        
        # Add to queue
        await self.message_queue.put(message)
    
    def perceive(self, message: Message) -> Dict[str, Any]:
        """
        Process an incoming message to extract relevant information
        
        Args:
            message: The message to perceive
            
        Returns:
            Dictionary of perceived information
        """
        # Extract relevant information based on message type
        perception = {
            "message_id": message.id,
            "sender": message.sender,
            "message_type": message.message_type,
            "timestamp": message.timestamp,
            "content": message.content,
            "thread_id": message.thread_id,
        }
        
        # Enhance perception with contextual information
        if message.message_type == MessageType.QUERY:
            # Retrieve relevant context for queries
            query = message.content.get("query", "")
            if query:
                relevant_context = self.memory.get_relevant_context(query)
                perception["relevant_context"] = relevant_context
        
        return perception
    
    @abstractmethod
    async def think(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate thoughts based on perception
        
        Args:
            perception: Perceived information
            
        Returns:
            Dictionary containing thoughts and reasoning
        """
        pass
    
    @abstractmethod
    async def decide_action(self, thoughts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decide what action to take based on thoughts
        
        Args:
            thoughts: The agent's thoughts
            
        Returns:
            Dictionary containing action decision
        """
        pass
    
    async def act(self, action_decision: Dict[str, Any]) -> Any:
        """
        Execute the decided action
        
        Args:
            action_decision: The action decision
            
        Returns:
            Result of the action
        """
        action_type = action_decision.get("action_type")
        
        if action_type == "send_message":
            return await self._send_message(action_decision)
        elif action_type == "use_tool":
            return await self._use_tool(action_decision)
        elif action_type == "update_status":
            return self._update_status(action_decision)
        elif action_type == "no_action":
            return None
        else:
            logger.warning(f"Unknown action type: {action_type}")
            return None
    
    async def _send_message(self, action_decision: Dict[str, Any]) -> Message:
        """
        Send a message to another agent or the environment
        
        Args:
            action_decision: Decision containing message details
            
        Returns:
            The sent message
        """
        content = action_decision.get("content", {})
        recipient = action_decision.get("recipient")
        message_type = MessageType(action_decision.get("message_type", "chat"))
        thread_id = action_decision.get("thread_id")
        in_reply_to = action_decision.get("in_reply_to")
        
        message = Message(
            sender=self.name,
            recipient=recipient,
            message_type=message_type,
            content=content,
            thread_id=thread_id,
            in_reply_to=in_reply_to
        )
        
        # Update metrics
        self.metrics["messages_sent"] += 1
        
        logger.debug(f"Agent {self.name} sending message: {message.id} to {recipient or 'all'}")
        
        # Message will be dispatched by the environment
        return message
    
    async def _use_tool(self, action_decision: Dict[str, Any]) -> Any:
        """
        Use a tool to perform an action
        
        Args:
            action_decision: Decision containing tool details
            
        Returns:
            Result of using the tool
        """
        tool_name = action_decision.get("tool_name")
        args = action_decision.get("args", [])
        kwargs = action_decision.get("kwargs", {})
        
        if tool_name not in self.tools:
            logger.warning(f"Agent {self.name} attempted to use unknown tool: {tool_name}")
            return {"error": f"Unknown tool: {tool_name}"}
        
        tool = self.tools[tool_name]
        
        try:
            # Update metrics
            self.metrics["tools_used"] += 1
            
            logger.debug(f"Agent {self.name} using tool: {tool_name}")
            
            # Execute the tool
            result = tool.execute(*args, **kwargs)
            return {"result": result, "tool": tool_name}
        except Exception as e:
            logger.error(f"Error using tool {tool_name}: {str(e)}")
            self.metrics["errors_encountered"] += 1
            return {"error": str(e), "tool": tool_name}
    
    def _update_status(self, action_decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the agent's status
        
        Args:
            action_decision: Decision containing status details
            
        Returns:
            Updated status information
        """
        new_status = action_decision.get("status")
        if new_status in [status.value for status in AgentStatus]:
            self.status = AgentStatus(new_status)
            logger.info(f"Agent {self.name} updated status to {self.status.value}")
            return {"status": self.status.value}
        else:
            logger.warning(f"Agent {self.name} attempted to set invalid status: {new_status}")
            return {"error": f"Invalid status: {new_status}", "current_status": self.status.value}
    
    async def run_perception_cycle(self) -> None:
        """
        Run a single perception-cognition-action cycle
        
        This method implements the basic cognitive loop:
        1. Get a message from the queue
        2. Perceive the message
        3. Think about the perception
        4. Decide what action to take
        5. Execute the action
        """
        try:
            # Get a message from the queue
            message = await self.message_queue.get()
            
            start_time = datetime.datetime.now()
            
            # Perceive the message
            perception = self.perceive(message)
            
            # Think about the perception
            thoughts = await self.think(perception)
            
            # Decide what action to take
            action_decision = await self.decide_action(thoughts)
            
            # Execute the action
            result = await self.act(action_decision)
            
            end_time = datetime.datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            self.metrics["response_time"].append(processing_time)
            
            # Mark task as done
            self.message_queue.task_done()
            
            # Return the result (will be handled by the environment)
            return result
        except Exception as e:
            logger.error(f"Error in perception cycle for agent {self.name}: {str(e)}")
            self.metrics["errors_encountered"] += 1
            return None
    
    async def run(self) -> None:
        """
        Run the agent continuously
        
        Continuously processes messages and executes perception cycles
        """
        self.status = AgentStatus.ACTIVE
        logger.info(f"Agent {self.name} starting continuous operation")
        
        try:
            while self.status == AgentStatus.ACTIVE:
                # Process one message
                result = await self.run_perception_cycle()
                
                # If we need to wait for more messages, add a small delay
                if self.message_queue.empty():
                    await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"Error in agent {self.name} main loop: {str(e)}")
            self.status = AgentStatus.ERROR
        finally:
            # Save agent memory before shutting down
            self.memory.save()
            logger.info(f"Agent {self.name} shutting down, memory saved")
            
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get agent performance metrics
        
        Returns:
            Dictionary of performance metrics
        """
        metrics = self.metrics.copy()
        
        # Calculate average response time
        if metrics["response_time"]:
            metrics["avg_response_time"] = sum(metrics["response_time"]) / len(metrics["response_time"])
        else:
            metrics["avg_response_time"] = 0
            
        return metrics


class LLMAgent(Agent):
    """
    Agent implementation using a Language Model for cognition
    
    This class implements the abstract think() and decide_action() methods
    using prompts to the LLM for generating thoughts and decisions.
    """
    
    async def think(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate thoughts using the LLM based on perception
        
        Args:
            perception: Perceived information
            
        Returns:
            Dictionary containing thoughts and reasoning
        """
        # Format perception for the prompt
        perception_text = json.dumps(perception, indent=2)
        
        # Get relevant context from memory
        recent_messages = self.memory.get_recent_messages(5)
        recent_messages_text = "\n".join([
            f"From: {m.sender}, To: {m.recipient or 'All'}, Type: {m.message_type.value}, "
            f"Content: {json.dumps(m.content)}"
            for m in recent_messages
        ])
        
        # Create the prompt for thinking
        thinking_template = PromptTemplate(
            input_variables=["system_prompt", "agent_name", "agent_role", "perception", "recent_messages"],
            template="""
            {system_prompt}
            
            You are currently processing information as {agent_name}, a {agent_role}.
            
            ## Recent Messages:
            {recent_messages}
            
            ## Current Perception:
            {perception}
            
            ## Think:
            Analyze the situation and generate your thoughts. Consider:
            1. What is the key information in this message?
            2. How does it relate to your goals and role?
            3. What context from memory is relevant?
            4. What are the implications?
            
            Provide your thoughts in a structured format.
            """
        )
        
        # Generate thoughts using the LLM
        thinking_prompt = thinking_template.format(
            system_prompt=self.system_prompt,
            agent_name=self.name,
            agent_role=self.role,
            perception=perception_text,
            recent_messages=recent_messages_text
        )
        
        try:
            thoughts_text = await self.llm.agenerate([thinking_prompt], temperature=self.llm_temperature, max_tokens=self.llm_max_tokens)
            thoughts_str = thoughts_text.generations[0][0].text.strip()
            
            # Parse thoughts into a structured format
            thoughts = {
                "raw": thoughts_str,
                "timestamp": datetime.datetime.now().timestamp(),
                "perception_id": perception.get("message_id")
            }
            
            logger.debug(f"Agent {self.name} generated thoughts for message {perception.get('message_id')}")
            
            return thoughts
        except Exception as e:
            logger.error(f"Error generating thoughts for agent {self.name}: {str(e)}")
            self.metrics["errors_encountered"] += 1
            return {
                "raw": f"Error generating thoughts: {str(e)}",
                "timestamp": datetime.datetime.now().timestamp(),
                "perception_id": perception.get("message_id"),
                "error": str(e)
            }
    
    async def decide_action(self, thoughts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decide what action to take using the LLM based on thoughts
        
        Args:
            thoughts: The agent's thoughts
            
        Returns:
            Dictionary containing action decision
        """
        # Format thoughts for the prompt
        thoughts_text = thoughts.get("raw", "")
        
        # Get available tools
        tools_text = "\n".join([
            f"- {name}: {tool.description}" for name, tool in self.tools.items()
        ])
        
        # Create the prompt for decision making
        decision_template = PromptTemplate(
            input_variables=["system_prompt", "agent_name", "agent_role", "thoughts", "tools"],
            template="""
            {system_prompt}
            
            You are currently deciding on actions as {agent_name}, a {agent_role}.
            
            ## Your Thoughts:
            {thoughts}
            
            ## Available Tools:
            {tools}
            
            ## Action Decision:
            Based on your thoughts, decide what action to take. Your options include:
            
            1. send_message: Send a message to another agent or broadcast to all
            2. use_tool: Use one of your available tools
            3. update_status: Update your operational status
            4. no_action: Take no action
            
            Provide your decision in a structured JSON format with the following schema:
            
            For send_message:
            {{
                "action_type": "send_message",
                "recipient": "<recipient_name or null for broadcast>",
                "message_type": "<chat|command|query|response|status>",
                "content": {{
                    "text": "<message_text>",
                    "additional_data": "<any_additional_data>"
                }},
                "thread_id": "<thread_id if continuing a conversation>",
                "in_reply_to": "<message_id if replying to a specific message>"
            }}
            
            For use_tool:
            {{
                "action_type": "use_tool",
                "tool_name": "<name_of_tool>",
                "args": [<positional_arguments>],
                "kwargs": {{<keyword_arguments>}}
            }}
            
            For update_status:
            {{
                "action_type": "update_status",
                "status": "<active|paused|terminated>"
            }}
            
            For no_action:
            {{
                "action_type": "no_action",
                "reason": "<reason_for_no_action>"
            }}
            
            Only output the JSON object with no additional text.
            """
        )
        
        # Generate decision using the LLM
        decision_prompt = decision_template.format(
            system_prompt=self.system_prompt,
            agent_name=self.name,
            agent_role=self.role,
            thoughts=thoughts_text,
            tools=tools_text
        )
        
        try:
            decision_text = await self.llm.agenerate([decision_prompt], temperature=self.llm_temperature, max_tokens=self.llm_max_tokens)
            decision_str = decision_text.generations[0][0].text.strip()
            
            # Extract JSON from the response
            try:
                # Find JSON in the response - sometimes LLM adds explanation text
                import re
                json_match = re.search(r'({.*})', decision_str, re.DOTALL)
                if json_match:
                    decision_str = json_match.group(1)
                
                decision = json.loads(decision_str)
                logger.debug(f"Agent {self.name} decided on action: {decision.get('action_type')}")
                return decision
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing decision JSON for agent {self.name}: {str(e)}")
                logger.error(f"Raw decision: {decision_str}")
                # Fall back to no_action
                return {
                    "action_type": "no_action",
                    "reason": f"Error parsing decision: {str(e)}"
                }
        except Exception as e:
            logger.error(f"Error generating decision for agent {self.name}: {str(e)}")
            self.metrics["errors_encountered"] += 1
            return {
                "action_type": "no_action",
                "reason": f"Error generating decision: {str(e)}"
            }


# Project Manager specific agent
class ProjectManagerAgent(LLMAgent):
    """
    Specialized agent for project management
    
    This agent coordinates other agents, tracks progress, and ensures
    the team is moving toward its goals.
    """
    
    def __init__(
        self,
        config: AgentConfig,
        llm: BaseLLM,
        callback_manager: Optional[CallbackManager] = None,
    ):
        super().__init__(config, llm, callback_manager)
        
        # Project management specific state
        self.project_state = {
            "goals": [],
            "tasks": {},
            "team_members": {},
            "progress": {},
            "issues": [],
            "timeline": {}
        }
        
        # Initialize PM-specific memory keys
        self.memory.set_working_memory("project_state", self.project_state)
        
        logger.info(f"Initialized ProjectManagerAgent: {self.name}")
    
    def set_project_goals(self, goals: List[str]) -> None:
        """
        Set the goals for the project
        
        Args:
            goals: List of project goals
        """
        self.project_state["goals"] = goals
        self.memory.set_working_memory("project_state", self.project_state)
        logger.info(f"ProjectManagerAgent {self.name} set goals: {goals}")
    
    def add_team_member(self, agent_name: str, agent_role: str) -> None:
        """
        Add a team member to the project
        
        Args:
            agent_name: Name of the agent
            agent_role: Role of the agent
        """
        self.project_state["team_members"][agent_name] = {
            "role": agent_role,
            "status": "active",
            "tasks_assigned": [],
            "tasks_completed": []
        }
        self.memory.set_working_memory("project_state", self.project_state)
        logger.info(f"ProjectManagerAgent {self.name} added team member: {agent_name} ({agent_role})")
    
    def assign_task(self, task_id: str, task_description: str, assignee: str, deadline: Optional[str] = None) -> None:
        """
        Assign a task to a team member
        
        Args:
            task_id: Unique ID for the task
            task_description: Description of the task
            assignee: Name of the agent assigned to the task
            deadline: Optional deadline for the task
        """
        if assignee not in self.project_state["team_members"]:
            logger.warning(f"Cannot assign task to unknown team member: {assignee}")
            return
        
        self.project_state["tasks"][task_id] = {
            "description": task_description,
            "assignee": assignee,
            "status": "assigned",
            "deadline": deadline,
            "created_at": datetime.datetime.now().isoformat(),
            "updated_at": datetime.datetime.now().isoformat(),
            "updates": []
        }
        
        self.project_state["team_members"][assignee]["tasks_assigned"].append(task_id)
        self.memory.set_working_memory("project_state", self.project_state)
        logger.info(f"ProjectManagerAgent {self.name} assigned task {task_id} to {assignee}")
    
    def update_task_status(self, task_id: str, status: str, update_message: Optional[str] = None) -> None:
        """
        Update the status of a task
        
        Args:
            task_id: ID of the task to update
            status: New status for the task
            update_message: Optional message describing the update
        """
        if task_id not in self.project_state["tasks"]:
            logger.warning(f"Cannot update unknown task: {task_id}")
            return
        
        task = self.project_state["tasks"][task_id]
        task["status"] = status
        task["updated_at"] = datetime.datetime.now().isoformat()
        
        if update_message:
            task["updates"].append({
                "timestamp": datetime.datetime.now().isoformat(),
                "message": update_message
            })
        
        if status == "completed":
            assignee = task["assignee"]
            if assignee in self.project_state["team_members"]:
                if task_id in self.project_state["team_members"][assignee]["tasks_assigned"]:
                    self.project_state["team_members"][assignee]["tasks_assigned"].remove(task_id)
                self.project_state["team_members"][assignee]["tasks_completed"].append(task_id)
        
        self.memory.set_working_memory("project_state", self.project_state)
        logger.info(f"ProjectManagerAgent {self.name} updated task {task_id} status to {status}")
    
    def get_project_status(self) -> Dict[str, Any]:
        """
        Get the current status of the project
        
        Returns:
            Dictionary containing project status
        """
        # Calculate progress metrics
        total_tasks = len(self.project_state["tasks"])
        completed_tasks = sum(1 for task in self.project_state["tasks"].values() if task["status"] == "completed")
        in_progress_tasks = sum(1 for task in self.project_state["tasks"].values() if task["status"] in ["in_progress", "reviewing"])
        
        progress_percentage = 0
        if total_tasks > 0:
            progress_percentage = (completed_tasks / total_tasks) * 100
        
        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "in_progress_tasks": in_progress_tasks,
            "pending_tasks": total_tasks - completed_tasks - in_progress_tasks,
            "progress_percentage": progress_percentage,
            "team_size": len(self.project_state["team_members"]),
            "issues": len(self.project_state["issues"])
        }
    
    async def think(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """
        Override base think method to include project management context
        
        Args:
            perception: Perceived information
            
        Returns:
            Dictionary containing thoughts and reasoning
        """
        # Get project status
        project_status = self.get_project_status()
        
        # Format perception for the prompt
        perception_text = json.dumps(perception, indent=2)
        
        # Get relevant context from memory
        recent_messages = self.memory.get_recent_messages(5)
        recent_messages_text = "\n".join([
            f"From: {m.sender}, To: {m.recipient or 'All'}, Type: {m.message_type.value}, "
            f"Content: {json.dumps(m.content)}"
            for m in recent_messages
        ])
        
        # Format project state
        project_state_text = json.dumps(self.project_state, indent=2)
        
        # Create the prompt for thinking
        thinking_template = PromptTemplate(
            input_variables=[
                "system_prompt", "agent_name", "perception", 
                "recent_messages", "project_state", "project_status"
            ],
            template="""
            {system_prompt}
            
            You are currently processing information as {agent_name}, the Project Manager.
            
            ## Project Status:
            {project_status}
            
            ## Recent Messages:
            {recent_messages}
            
            ## Current Perception:
            {perception}
            
            ## Project State:
            {project_state}
            
            ## Think:
            As the Project Manager, analyze the situation considering:
            1. What is the key information in this message?
            2. How does it relate to the project goals and current state?
            3. Does this require task assignment or status updates?
            4. Are there any issues or risks that need to be addressed?
            5. How can you keep the team aligned and productive?
            
            Provide your thoughts in a structured format.
            """
        )
        
        # Generate thoughts using the LLM
        thinking_prompt = thinking_template.format(
            system_prompt=self.system_prompt,
            agent_name=self.name,
            perception=perception_text,
            recent_messages=recent_messages_text,
            project_state=project_state_text,
            project_status=json.dumps(project_status, indent=2)
        )
        
        try:
            thoughts_text = await self.llm.agenerate([thinking_prompt], temperature=self.llm_temperature, max_tokens=self.llm_max_tokens)
            thoughts_str = thoughts_text.generations[0][0].text.strip()
            
            # Parse thoughts into a structured format
            thoughts = {
                "raw": thoughts_str,
                "timestamp": datetime.datetime.now().timestamp(),
                "perception_id": perception.get("message_id"),
                "project_status": project_status
            }
            
            logger.debug(f"ProjectManagerAgent {self.name} generated thoughts for message {perception.get('message_id')}")
            
            return thoughts
        except Exception as e:
            logger.error(f"Error generating thoughts for ProjectManagerAgent {self.name}: {str(e)}")
            self.metrics["errors_encountered"] += 1
            return {
                "raw": f"Error generating thoughts: {str(e)}",
                "timestamp": datetime.datetime.now().timestamp(),
                "perception_id": perception.get("message_id"),
                "error": str(e),
                "project_status": project_status
            }
    
    async def decide_action(self, thoughts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Override base decide_action method to include project management options
        
        Args:
            thoughts: The agent's thoughts
            
        Returns:
            Dictionary containing action decision
        """
        # Format thoughts for the prompt
        thoughts_text = thoughts.get("raw", "")
        
        # Get available tools
        tools_text = "\n".join([
            f"- {name}: {tool.description}" for name, tool in self.tools.items()
        ])
        
        # Format project status
        project_status = thoughts.get("project_status", {})
        project_status_text = json.dumps(project_status, indent=2)
        
        # Create the prompt for decision making
        decision_template = PromptTemplate(
            input_variables=[
                "system_prompt", "agent_name", "thoughts", 
                "tools", "project_status"
            ],
            template="""
            {system_prompt}
            
            You are currently deciding on actions as {agent_name}, the Project Manager.
            
            ## Project Status:
            {project_status}
            
            ## Your Thoughts:
            {thoughts}
            
            ## Available Tools:
            {tools}
            
            ## Action Decision:
            Based on your thoughts, decide what action to take. Your options include:
            
            1. send_message: Send a message to another agent or broadcast to all
            2. use_tool: Use one of your available tools
            3. project_action: Take a project management action (assign task, update status, etc.)
            4. update_status: Update your operational status
            5. no_action: Take no action
            
            Provide your decision in a structured JSON format with the following schema:
            
            For send_message:
            {{
                "action_type": "send_message",
                "recipient": "<recipient_name or null for broadcast>",
                "message_type": "<chat|command|query|response|status>",
                "content": {{
                    "text": "<message_text>",
                    "additional_data": "<any_additional_data>"
                }},
                "thread_id": "<thread_id if continuing a conversation>",
                "in_reply_to": "<message_id if replying to a specific message>"
            }}
            
            For use_tool:
            {{
                "action_type": "use_tool",
                "tool_name": "<name_of_tool>",
                "args": [<positional_arguments>],
                "kwargs": {{<keyword_arguments>}}
            }}
            
            For project_action:
            {{
                "action_type": "project_action",
                "project_action": "<assign_task|update_task|add_team_member|set_goals>",
                "action_data": {{
                    // Data specific to the project action
                }}
            }}
            
            For update_status:
            {{
                "action_type": "update_status",
                "status": "<active|paused|terminated>"
            }}
            
            For no_action:
            {{
                "action_type": "no_action",
                "reason": "<reason_for_no_action>"
            }}
            
            Only output the JSON object with no additional text.
            """
        )
        
        # Generate decision using the LLM
        decision_prompt = decision_template.format(
            system_prompt=self.system_prompt,
            agent_name=self.name,
            thoughts=thoughts_text,
            tools=tools_text,
            project_status=project_status_text
        )
        
        try:
            decision_text = await self.llm.agenerate([decision_prompt], temperature=self.llm_temperature, max_tokens=self.llm_max_tokens)
            decision_str = decision_text.generations[0][0].text.strip()
            
            # Extract JSON from the response
            try:
                # Find JSON in the response
                import re
                json_match = re.search(r'({.*})', decision_str, re.DOTALL)
                if json_match:
                    decision_str = json_match.group(1)
                
                decision = json.loads(decision_str)
                
                # Handle project-specific actions
                if decision.get("action_type") == "project_action":
                    decision = await self._handle_project_action(decision)
                
                logger.debug(f"ProjectManagerAgent {self.name} decided on action: {decision.get('action_type')}")
                return decision
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing decision JSON for ProjectManagerAgent {self.name}: {str(e)}")
                logger.error(f"Raw decision: {decision_str}")
                # Fall back to no_action
                return {
                    "action_type": "no_action",
                    "reason": f"Error parsing decision: {str(e)}"
                }
        except Exception as e:
            logger.error(f"Error generating decision for ProjectManagerAgent {self.name}: {str(e)}")
            self.metrics["errors_encountered"] += 1
            return {
                "action_type": "no_action",
                "reason": f"Error generating decision: {str(e)}"
            }
    
    async def _handle_project_action(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle project-specific actions
        
        Args:
            decision: The project action decision
            
        Returns:
            Updated action decision for the agent system
        """
        project_action = decision.get("project_action")
        action_data = decision.get("action_data", {})
        
        if project_action == "assign_task":
            task_id = action_data.get("task_id", str(uuid.uuid4()))
            task_description = action_data.get("description", "")
            assignee = action_data.get("assignee", "")
            deadline = action_data.get("deadline")
            
            self.assign_task(task_id, task_description, assignee, deadline)
            
            # Convert to send_message action to notify the assignee
            return {
                "action_type": "send_message",
                "recipient": assignee,
                "message_type": MessageType.COMMAND.value,
                "content": {
                    "text": f"You have been assigned a new task: {task_description}",
                    "task_id": task_id,
                    "deadline": deadline
                }
            }
        
        elif project_action == "update_task":
            task_id = action_data.get("task_id", "")
            status = action_data.get("status", "")
            update_message = action_data.get("message", "")
            
            self.update_task_status(task_id, status, update_message)
            
            # Convert to a broadcast message to notify the team
            return {
                "action_type": "send_message",
                "recipient": None,  # Broadcast
                "message_type": MessageType.STATUS.value,
                "content": {
                    "text": f"Task {task_id} status updated to: {status}",
                    "task_id": task_id,
                    "status": status,
                    "message": update_message
                }
            }
        
        elif project_action == "add_team_member":
            agent_name = action_data.get("name", "")
            agent_role = action_data.get("role", "")
            
            self.add_team_member(agent_name, agent_role)
            
            # Convert to a broadcast message to notify the team
            return {
                "action_type": "send_message",
                "recipient": None,  # Broadcast
                "message_type": MessageType.SYSTEM.value,
                "content": {
                    "text": f"New team member added: {agent_name} ({agent_role})",
                    "agent_name": agent_name,
                    "agent_role": agent_role
                }
            }
        
        elif project_action == "set_goals":
            goals = action_data.get("goals", [])
            
            self.set_project_goals(goals)
            
            # Convert to a broadcast message to notify the team
            return {
                "action_type": "send_message",
                "recipient": None,  # Broadcast
                "message_type": MessageType.SYSTEM.value,
                "content": {
                    "text": f"Project goals have been set: {', '.join(goals)}",
                    "goals": goals
                }
            }
        
        else:
            logger.warning(f"Unknown project action: {project_action}")
            return {
                "action_type": "no_action",
                "reason": f"Unknown project action: {project_action}"
            }


# Factory function to create specialized agents
def create_agent(
    config: AgentConfig,
    llm: BaseLLM,
    callback_manager: Optional[CallbackManager] = None
) -> Agent:
    """
    Factory function to create the appropriate agent based on role
    
    Args:
        config: Configuration for the agent
        llm: Language model to use
        callback_manager: Optional callback manager
        
    Returns:
        The created agent
    """
    role_lower = config.role.lower()
    
    if "project manager" in role_lower or "team lead" in role_lower:
        return ProjectManagerAgent(config, llm, callback_manager)
    else:
        return LLMAgent(config, llm, callback_manager)