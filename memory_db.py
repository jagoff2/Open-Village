"""
Offline Memory Database Module - Persistent storage for agent knowledge

This module provides a robust, persistent memory system for agents to store and
retrieve information, ensuring knowledge persists across sessions.

Dependencies:
- Python 3.10+
- pydantic 2.3+
- chromadb 0.4.6+
- SQLAlchemy 2.0+
- langchain 0.0.267+
"""

import asyncio
import datetime
import json
import logging
import os
import pickle
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable

import chromadb
import numpy as np
import torch
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, String, Float, Text, DateTime, Boolean, ForeignKey, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("memory_db.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# SQLAlchemy Base
Base = declarative_base()


class MemoryEntry(Base):
    """Database model for memory entries"""
    __tablename__ = "memory_entries"
    
    id = Column(String, primary_key=True)
    agent_id = Column(String, index=True)
    timestamp = Column(Float, index=True)
    memory_type = Column(String, index=True)
    content = Column(Text)
    metadata = Column(Text)
    importance = Column(Float, default=0.0, index=True)
    last_accessed = Column(Float, index=True)
    access_count = Column(Integer, default=0)
    embedding_id = Column(String, nullable=True)
    created_at = Column(Float, default=time.time)
    updated_at = Column(Float, default=time.time, onupdate=time.time)
    
    tags = relationship("MemoryTag", back_populates="memory")


class MemoryTag(Base):
    """Database model for memory tags"""
    __tablename__ = "memory_tags"
    
    id = Column(String, primary_key=True)
    memory_id = Column(String, ForeignKey("memory_entries.id"))
    tag = Column(String, index=True)
    
    memory = relationship("MemoryEntry", back_populates="tags")


class MemoryRelation(Base):
    """Database model for relationships between memories"""
    __tablename__ = "memory_relations"
    
    id = Column(String, primary_key=True)
    source_id = Column(String, ForeignKey("memory_entries.id"), index=True)
    target_id = Column(String, ForeignKey("memory_entries.id"), index=True)
    relation_type = Column(String, index=True)
    strength = Column(Float, default=0.0)
    metadata = Column(Text)
    created_at = Column(Float, default=time.time)
    updated_at = Column(Float, default=time.time, onupdate=time.time)


class MemoryConfig(BaseModel):
    """Configuration for memory database"""
    storage_path: str = "./memory_storage"
    embedding_dimension: int = 1536
    embedding_model: str = "openai"
    max_entries_per_agent: int = 10000
    cleanup_interval: int = 3600  # seconds
    custom_configuration: Dict[str, Any] = Field(default_factory=dict)


class MemoryType(Enum):
    """Types of memories that can be stored"""
    OBSERVATION = "observation"
    THOUGHT = "thought"
    CONVERSATION = "conversation"
    FACT = "fact"
    EXPERIENCE = "experience"
    REFLECTION = "reflection"
    PLAN = "plan"
    ACTION = "action"
    SKILL = "skill"
    PERSONALITY = "personality"
    VALUE = "value"
    GOAL = "goal"
    RESOURCE = "resource"


class MemoryDatabase:
    """
    Persistent memory database for agent knowledge
    
    This class provides a robust storage system for agent memories,
    allowing for efficient storage, retrieval, and querying of information.
    """
    def __init__(self, config: MemoryConfig):
        self.storage_path = config.storage_path
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Initialize SQLite database
        self.db_path = os.path.join(self.storage_path, "memory.db")
        self.engine = create_engine(f"sqlite:///{self.db_path}")
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        # Initialize vector database
        self.embedding_dimension = config.embedding_dimension
        self.chromadb_path = os.path.join(self.storage_path, "chromadb")
        os.makedirs(self.chromadb_path, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=self.chromadb_path)
        
        # Initialize embedding model
        if config.embedding_model == "openai":
            self.embedding_model = OpenAIEmbeddings()
        else:
            # Default to a simple embedding function for testing
            self._init_simple_embedding_model()
        
        # Create or get the main collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="memories",
            metadata={"dimension": self.embedding_dimension}
        )
        
        # Configuration
        self.max_entries_per_agent = config.max_entries_per_agent
        self.cleanup_interval = config.cleanup_interval
        self.custom_config = config.custom_configuration
        
        # Start cleanup task if interval is positive
        if self.cleanup_interval > 0:
            self._start_cleanup_task()
        
        logger.info(f"Initialized memory database at {self.storage_path}")
    
    def _init_simple_embedding_model(self) -> None:
        """Initialize a simple embedding model for testing"""
        def simple_embedding(texts: List[str]) -> List[List[float]]:
            # Generate random embeddings for testing
            embeddings = []
            for text in texts:
                # Use hash of text as seed for reproducibility
                np.random.seed(hash(text) % 2**32)
                embedding = np.random.rand(self.embedding_dimension)
                embedding = embedding / np.linalg.norm(embedding)  # Normalize
                embeddings.append(embedding.tolist())
            return embeddings
        
        self.embedding_model = simple_embedding
    
    def _start_cleanup_task(self) -> None:
        """Start a background task to periodically clean up old memories"""
        async def cleanup_task():
            while True:
                try:
                    # Sleep first to avoid immediate cleanup on startup
                    await asyncio.sleep(self.cleanup_interval)
                    self.cleanup_old_memories()
                except Exception as e:
                    logger.error(f"Error in memory cleanup task: {str(e)}")
        
        # Start the task
        asyncio.create_task(cleanup_task())
        logger.info(f"Started memory cleanup task with interval {self.cleanup_interval} seconds")
    
    def _compute_embedding(self, text: str) -> List[float]:
        """
        Compute embedding for a text string
        
        Args:
            text: Text to compute embedding for
            
        Returns:
            Embedding vector as a list of floats
        """
        if hasattr(self.embedding_model, "embed_query"):
            # LangChain compatible model
            return self.embedding_model.embed_query(text)
        else:
            # Custom embedding function
            return self.embedding_model([text])[0]
    
    def add_memory(self, agent_id: str, memory_type: MemoryType, 
                  content: str, metadata: Optional[Dict[str, Any]] = None,
                  tags: Optional[List[str]] = None,
                  importance: float = 0.0) -> str:
        """
        Add a new memory for an agent
        
        Args:
            agent_id: ID of the agent
            memory_type: Type of memory
            content: Text content of the memory
            metadata: Additional metadata for the memory
            tags: Tags for the memory
            importance: Importance score (0.0 to 1.0)
            
        Returns:
            ID of the new memory entry
        """
        # Generate a unique ID
        memory_id = str(uuid.uuid4())
        
        # Compute embedding
        embedding = self._compute_embedding(content)
        
        # Add to vector database
        self.collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            metadatas=[{
                "agent_id": agent_id,
                "memory_type": memory_type.value if isinstance(memory_type, MemoryType) else memory_type,
                "timestamp": time.time(),
                "importance": importance
            }],
            documents=[content]
        )
        
        # Add to SQL database
        session = self.Session()
        try:
            # Create the memory entry
            memory_entry = MemoryEntry(
                id=memory_id,
                agent_id=agent_id,
                timestamp=time.time(),
                memory_type=memory_type.value if isinstance(memory_type, MemoryType) else memory_type,
                content=content,
                metadata=json.dumps(metadata or {}),
                importance=importance,
                last_accessed=time.time(),
                embedding_id=memory_id
            )
            session.add(memory_entry)
            
            # Add tags if provided
            if tags:
                for tag in tags:
                    tag_entry = MemoryTag(
                        id=str(uuid.uuid4()),
                        memory_id=memory_id,
                        tag=tag
                    )
                    session.add(tag_entry)
            
            session.commit()
            logger.debug(f"Added memory {memory_id} for agent {agent_id}")
            return memory_id
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding memory for agent {agent_id}: {str(e)}")
            raise
        finally:
            session.close()
    
    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a memory by ID
        
        Args:
            memory_id: ID of the memory
            
        Returns:
            Dictionary containing memory data if found, None otherwise
        """
        session = self.Session()
        try:
            # Get the memory entry
            memory_entry = session.query(MemoryEntry).filter(MemoryEntry.id == memory_id).first()
            
            if not memory_entry:
                return None
            
            # Update access metrics
            memory_entry.last_accessed = time.time()
            memory_entry.access_count += 1
            session.commit()
            
            # Get tags
            tags = [tag.tag for tag in memory_entry.tags]
            
            # Compile result
            result = {
                "id": memory_entry.id,
                "agent_id": memory_entry.agent_id,
                "timestamp": memory_entry.timestamp,
                "memory_type": memory_entry.memory_type,
                "content": memory_entry.content,
                "metadata": json.loads(memory_entry.metadata),
                "importance": memory_entry.importance,
                "last_accessed": memory_entry.last_accessed,
                "access_count": memory_entry.access_count,
                "created_at": memory_entry.created_at,
                "updated_at": memory_entry.updated_at,
                "tags": tags
            }
            
            logger.debug(f"Retrieved memory {memory_id}")
            return result
        except Exception as e:
            logger.error(f"Error retrieving memory {memory_id}: {str(e)}")
            return None
        finally:
            session.close()
    
    def update_memory(self, memory_id: str, content: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None,
                     importance: Optional[float] = None,
                     tags: Optional[List[str]] = None) -> bool:
        """
        Update an existing memory
        
        Args:
            memory_id: ID of the memory to update
            content: New content (if provided)
            metadata: New metadata (if provided)
            importance: New importance score (if provided)
            tags: New tags (if provided)
            
        Returns:
            True if the update was successful, False otherwise
        """
        session = self.Session()
        try:
            # Get the memory entry
            memory_entry = session.query(MemoryEntry).filter(MemoryEntry.id == memory_id).first()
            
            if not memory_entry:
                logger.warning(f"Memory {memory_id} not found for update")
                return False
            
            # Update fields if provided
            if content is not None:
                memory_entry.content = content
                
                # Update embedding
                embedding = self._compute_embedding(content)
                self.collection.update(
                    ids=[memory_id],
                    embeddings=[embedding],
                    documents=[content]
                )
            
            if metadata is not None:
                # Merge with existing metadata
                existing_metadata = json.loads(memory_entry.metadata)
                existing_metadata.update(metadata)
                memory_entry.metadata = json.dumps(existing_metadata)
            
            if importance is not None:
                memory_entry.importance = importance
                
                # Update metadata in vector DB
                self.collection.update(
                    ids=[memory_id],
                    metadatas=[{
                        "agent_id": memory_entry.agent_id,
                        "memory_type": memory_entry.memory_type,
                        "timestamp": memory_entry.timestamp,
                        "importance": importance
                    }]
                )
            
            if tags is not None:
                # Remove existing tags
                session.query(MemoryTag).filter(MemoryTag.memory_id == memory_id).delete()
                
                # Add new tags
                for tag in tags:
                    tag_entry = MemoryTag(
                        id=str(uuid.uuid4()),
                        memory_id=memory_id,
                        tag=tag
                    )
                    session.add(tag_entry)
            
            memory_entry.updated_at = time.time()
            session.commit()
            logger.debug(f"Updated memory {memory_id}")
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating memory {memory_id}: {str(e)}")
            return False
        finally:
            session.close()
    
    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory
        
        Args:
            memory_id: ID of the memory to delete
            
        Returns:
            True if the deletion was successful, False otherwise
        """
        session = self.Session()
        try:
            # Delete from SQL database
            memory_entry = session.query(MemoryEntry).filter(MemoryEntry.id == memory_id).first()
            
            if not memory_entry:
                logger.warning(f"Memory {memory_id} not found for deletion")
                return False
            
            # Delete related tags
            session.query(MemoryTag).filter(MemoryTag.memory_id == memory_id).delete()
            
            # Delete related relations
            session.query(MemoryRelation).filter(
                (MemoryRelation.source_id == memory_id) | 
                (MemoryRelation.target_id == memory_id)
            ).delete()
            
            # Delete the memory entry
            session.delete(memory_entry)
            session.commit()
            
            # Delete from vector database
            try:
                self.collection.delete(ids=[memory_id])
            except Exception as e:
                logger.warning(f"Error deleting memory {memory_id} from vector database: {str(e)}")
            
            logger.debug(f"Deleted memory {memory_id}")
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting memory {memory_id}: {str(e)}")
            return False
        finally:
            session.close()
    
    def search_by_content(self, query: str, agent_id: Optional[str] = None,
                        memory_type: Optional[MemoryType] = None,
                        n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search memories by semantic similarity to query
        
        Args:
            query: Query text
            agent_id: Filter by agent ID (optional)
            memory_type: Filter by memory type (optional)
            n_results: Maximum number of results to return
            
        Returns:
            List of memory entries sorted by relevance
        """
        # Compute query embedding
        query_embedding = self._compute_embedding(query)
        
        # Build filter
        filter_dict = {}
        if agent_id:
            filter_dict["agent_id"] = agent_id
        if memory_type:
            filter_dict["memory_type"] = memory_type.value if isinstance(memory_type, MemoryType) else memory_type
        
        # Search vector database
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_dict if filter_dict else None
        )
        
        # Compile results
        memories = []
        if results["ids"] and len(results["ids"][0]) > 0:
            memory_ids = results["ids"][0]
            distances = results["distances"][0]
            
            # Get full data from SQL database
            for i, memory_id in enumerate(memory_ids):
                memory_data = self.get_memory(memory_id)
                if memory_data:
                    memory_data["distance"] = distances[i]
                    memories.append(memory_data)
        
        logger.debug(f"Searched for '{query}', found {len(memories)} results")
        return memories
    
    def search_by_metadata(self, metadata_filters: Dict[str, Any],
                         agent_id: Optional[str] = None,
                         memory_type: Optional[MemoryType] = None,
                         limit: int = 100) -> List[Dict[str, Any]]:
        """
        Search memories by metadata filters
        
        Args:
            metadata_filters: Dictionary of metadata field-value pairs to filter by
            agent_id: Filter by agent ID (optional)
            memory_type: Filter by memory type (optional)
            limit: Maximum number of results to return
            
        Returns:
            List of matching memory entries
        """
        session = self.Session()
        try:
            query = session.query(MemoryEntry)
            
            # Apply filters
            if agent_id:
                query = query.filter(MemoryEntry.agent_id == agent_id)
            
            if memory_type:
                query = query.filter(MemoryEntry.memory_type == (memory_type.value if isinstance(memory_type, MemoryType) else memory_type))
            
            # Apply metadata filters using LIKE queries (simple approach)
            for key, value in metadata_filters.items():
                query = query.filter(MemoryEntry.metadata.like(f'%"{key}": {json.dumps(value)}%'))
            
            # Execute query with limit
            entries = query.order_by(MemoryEntry.timestamp.desc()).limit(limit).all()
            
            # Compile results
            memories = []
            for entry in entries:
                # Get tags
                tags = [tag.tag for tag in entry.tags]
                
                memory_data = {
                    "id": entry.id,
                    "agent_id": entry.agent_id,
                    "timestamp": entry.timestamp,
                    "memory_type": entry.memory_type,
                    "content": entry.content,
                    "metadata": json.loads(entry.metadata),
                    "importance": entry.importance,
                    "last_accessed": entry.last_accessed,
                    "access_count": entry.access_count,
                    "created_at": entry.created_at,
                    "updated_at": entry.updated_at,
                    "tags": tags
                }
                memories.append(memory_data)
            
            logger.debug(f"Searched by metadata, found {len(memories)} results")
            return memories
        except Exception as e:
            logger.error(f"Error searching by metadata: {str(e)}")
            return []
        finally:
            session.close()
    
    def search_by_tags(self, tags: List[str], agent_id: Optional[str] = None,
                     memory_type: Optional[MemoryType] = None,
                     match_all: bool = False,
                     limit: int = 100) -> List[Dict[str, Any]]:
        """
        Search memories by tags
        
        Args:
            tags: List of tags to search for
            agent_id: Filter by agent ID (optional)
            memory_type: Filter by memory type (optional)
            match_all: If True, only return memories that have all tags, otherwise match any
            limit: Maximum number of results to return
            
        Returns:
            List of matching memory entries
        """
        session = self.Session()
        try:
            # Base query for memory entries
            query = session.query(MemoryEntry).join(MemoryTag, MemoryEntry.id == MemoryTag.memory_id)
            
            # Apply agent filter if provided
            if agent_id:
                query = query.filter(MemoryEntry.agent_id == agent_id)
            
            # Apply memory type filter if provided
            if memory_type:
                query = query.filter(MemoryEntry.memory_type == (memory_type.value if isinstance(memory_type, MemoryType) else memory_type))
            
            # Apply tag filters
            if match_all:
                # Match all tags (more complex query)
                for tag in tags:
                    subquery = session.query(MemoryTag.memory_id).filter(MemoryTag.tag == tag).subquery()
                    query = query.filter(MemoryEntry.id.in_(subquery))
            else:
                # Match any tag
                query = query.filter(MemoryTag.tag.in_(tags))
            
            # Get distinct memory entries
            query = query.distinct(MemoryEntry.id)
            
            # Apply ordering and limit
            entries = query.order_by(MemoryEntry.timestamp.desc()).limit(limit).all()
            
            # Compile results
            memories = []
            for entry in entries:
                # Get tags
                entry_tags = [tag.tag for tag in entry.tags]
                
                memory_data = {
                    "id": entry.id,
                    "agent_id": entry.agent_id,
                    "timestamp": entry.timestamp,
                    "memory_type": entry.memory_type,
                    "content": entry.content,
                    "metadata": json.loads(entry.metadata),
                    "importance": entry.importance,
                    "last_accessed": entry.last_accessed,
                    "access_count": entry.access_count,
                    "created_at": entry.created_at,
                    "updated_at": entry.updated_at,
                    "tags": entry_tags
                }
                memories.append(memory_data)
            
            logger.debug(f"Searched by tags, found {len(memories)} results")
            return memories
        except Exception as e:
            logger.error(f"Error searching by tags: {str(e)}")
            return []
        finally:
            session.close()
    
    def get_agent_memories(self, agent_id: str, memory_type: Optional[MemoryType] = None,
                         order_by: str = "timestamp",
                         order_direction: str = "desc",
                         limit: int = 100,
                         offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get memories for a specific agent
        
        Args:
            agent_id: ID of the agent
            memory_type: Filter by memory type (optional)
            order_by: Field to order by (timestamp, importance, last_accessed)
            order_direction: Order direction (asc, desc)
            limit: Maximum number of results to return
            offset: Offset for pagination
            
        Returns:
            List of memory entries
        """
        session = self.Session()
        try:
            query = session.query(MemoryEntry).filter(MemoryEntry.agent_id == agent_id)
            
            # Apply memory type filter if provided
            if memory_type:
                query = query.filter(MemoryEntry.memory_type == (memory_type.value if isinstance(memory_type, MemoryType) else memory_type))
            
            # Apply ordering
            if order_by == "importance":
                order_column = MemoryEntry.importance
            elif order_by == "last_accessed":
                order_column = MemoryEntry.last_accessed
            else:  # Default to timestamp
                order_column = MemoryEntry.timestamp
            
            if order_direction.lower() == "asc":
                query = query.order_by(order_column.asc())
            else:
                query = query.order_by(order_column.desc())
            
            # Apply pagination
            entries = query.limit(limit).offset(offset).all()
            
            # Compile results
            memories = []
            for entry in entries:
                # Get tags
                tags = [tag.tag for tag in entry.tags]
                
                memory_data = {
                    "id": entry.id,
                    "agent_id": entry.agent_id,
                    "timestamp": entry.timestamp,
                    "memory_type": entry.memory_type,
                    "content": entry.content,
                    "metadata": json.loads(entry.metadata),
                    "importance": entry.importance,
                    "last_accessed": entry.last_accessed,
                    "access_count": entry.access_count,
                    "created_at": entry.created_at,
                    "updated_at": entry.updated_at,
                    "tags": tags
                }
                memories.append(memory_data)
            
            logger.debug(f"Retrieved {len(memories)} memories for agent {agent_id}")
            return memories
        except Exception as e:
            logger.error(f"Error retrieving memories for agent {agent_id}: {str(e)}")
            return []
        finally:
            session.close()
    
    def add_relation(self, source_id: str, target_id: str, relation_type: str,
                   strength: float = 0.5, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a relation between two memories
        
        Args:
            source_id: ID of the source memory
            target_id: ID of the target memory
            relation_type: Type of relation
            strength: Strength of the relation (0.0 to 1.0)
            metadata: Additional metadata for the relation
            
        Returns:
            ID of the new relation
        """
        session = self.Session()
        try:
            # Check if both memories exist
            source = session.query(MemoryEntry).filter(MemoryEntry.id == source_id).first()
            target = session.query(MemoryEntry).filter(MemoryEntry.id == target_id).first()
            
            if not source or not target:
                logger.warning(f"Cannot create relation: source or target memory not found")
                return None
            
            # Generate a unique ID
            relation_id = str(uuid.uuid4())
            
            # Create the relation
            relation = MemoryRelation(
                id=relation_id,
                source_id=source_id,
                target_id=target_id,
                relation_type=relation_type,
                strength=strength,
                metadata=json.dumps(metadata or {})
            )
            
            session.add(relation)
            session.commit()
            
            logger.debug(f"Added relation {relation_id} between {source_id} and {target_id}")
            return relation_id
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding relation: {str(e)}")
            return None
        finally:
            session.close()
    
    def get_related_memories(self, memory_id: str, relation_type: Optional[str] = None,
                           min_strength: float = 0.0, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get memories related to a specific memory
        
        Args:
            memory_id: ID of the memory
            relation_type: Filter by relation type (optional)
            min_strength: Minimum relation strength
            limit: Maximum number of results to return
            
        Returns:
            List of related memory entries
        """
        session = self.Session()
        try:
            # Query for relations where memory_id is source
            source_query = session.query(MemoryRelation).filter(
                MemoryRelation.source_id == memory_id,
                MemoryRelation.strength >= min_strength
            )
            
            # Query for relations where memory_id is target
            target_query = session.query(MemoryRelation).filter(
                MemoryRelation.target_id == memory_id,
                MemoryRelation.strength >= min_strength
            )
            
            # Apply relation type filter if provided
            if relation_type:
                source_query = source_query.filter(MemoryRelation.relation_type == relation_type)
                target_query = target_query.filter(MemoryRelation.relation_type == relation_type)
            
            # Get all relations
            source_relations = source_query.all()
            target_relations = target_query.all()
            
            # Combine results
            related_memories = []
            
            # Process source relations (memory_id -> target)
            for relation in source_relations:
                target_memory = self.get_memory(relation.target_id)
                if target_memory:
                    related_memories.append({
                        "memory": target_memory,
                        "relation": {
                            "id": relation.id,
                            "source_id": relation.source_id,
                            "target_id": relation.target_id,
                            "relation_type": relation.relation_type,
                            "strength": relation.strength,
                            "metadata": json.loads(relation.metadata),
                            "direction": "outgoing"
                        }
                    })
            
            # Process target relations (source -> memory_id)
            for relation in target_relations:
                source_memory = self.get_memory(relation.source_id)
                if source_memory:
                    related_memories.append({
                        "memory": source_memory,
                        "relation": {
                            "id": relation.id,
                            "source_id": relation.source_id,
                            "target_id": relation.target_id,
                            "relation_type": relation.relation_type,
                            "strength": relation.strength,
                            "metadata": json.loads(relation.metadata),
                            "direction": "incoming"
                        }
                    })
            
            # Sort by relation strength and limit results
            related_memories.sort(key=lambda x: x["relation"]["strength"], reverse=True)
            related_memories = related_memories[:limit]
            
            logger.debug(f"Retrieved {len(related_memories)} related memories for {memory_id}")
            return related_memories
        except Exception as e:
            logger.error(f"Error retrieving related memories for {memory_id}: {str(e)}")
            return []
        finally:
            session.close()
    
    def cleanup_old_memories(self, agent_id: Optional[str] = None, 
                           max_days: int = 30,
                           importance_threshold: float = 0.3) -> int:
        """
        Clean up old, low-importance memories
        
        Args:
            agent_id: Limit cleanup to a specific agent (optional)
            max_days: Maximum age in days for low-importance memories
            importance_threshold: Memories with importance below this are eligible for cleanup
            
        Returns:
            Number of memories deleted
        """
        session = self.Session()
        try:
            # Calculate cutoff timestamp
            cutoff_time = time.time() - (max_days * 24 * 60 * 60)
            
            # Base query for old, low-importance memories
            query = session.query(MemoryEntry).filter(
                MemoryEntry.timestamp < cutoff_time,
                MemoryEntry.importance < importance_threshold
            )
            
            # Apply agent filter if provided
            if agent_id:
                query = query.filter(MemoryEntry.agent_id == agent_id)
            
            # Get memories to delete
            memories_to_delete = query.all()
            
            # Delete each memory
            deleted_count = 0
            for memory in memories_to_delete:
                if self.delete_memory(memory.id):
                    deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} old memories")
            return deleted_count
        except Exception as e:
            session.rollback()
            logger.error(f"Error cleaning up old memories: {str(e)}")
            return 0
        finally:
            session.close()
    
    def get_stats(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics about the memory database
        
        Args:
            agent_id: Get stats for a specific agent (optional)
            
        Returns:
            Dictionary containing statistics
        """
        session = self.Session()
        try:
            stats = {}
            
            # Base queries
            total_query = session.query(MemoryEntry)
            type_counts_query = session.query(
                MemoryEntry.memory_type, 
                session.func.count(MemoryEntry.id)
            ).group_by(MemoryEntry.memory_type)
            
            # Apply agent filter if provided
            if agent_id:
                total_query = total_query.filter(MemoryEntry.agent_id == agent_id)
                type_counts_query = type_counts_query.filter(MemoryEntry.agent_id == agent_id)
                
                # Get agent-specific stats
                stats["agent_id"] = agent_id
            
            # Get total count
            total_count = total_query.count()
            stats["total_memories"] = total_count
            
            # Get counts by memory type
            type_counts = {}
            for memory_type, count in type_counts_query.all():
                type_counts[memory_type] = count
            stats["memory_types"] = type_counts
            
            # Get tag statistics
            tag_counts = {}
            tag_query = session.query(
                MemoryTag.tag,
                session.func.count(MemoryTag.id)
            ).group_by(MemoryTag.tag)
            
            if agent_id:
                tag_query = tag_query.join(MemoryEntry, MemoryTag.memory_id == MemoryEntry.id).filter(
                    MemoryEntry.agent_id == agent_id
                )
            
            for tag, count in tag_query.all():
                tag_counts[tag] = count
            stats["tags"] = tag_counts
            
            # Get relation statistics
            relation_counts = {}
            relation_query = session.query(
                MemoryRelation.relation_type,
                session.func.count(MemoryRelation.id)
            ).group_by(MemoryRelation.relation_type)
            
            if agent_id:
                relation_query = relation_query.join(
                    MemoryEntry, 
                    (MemoryRelation.source_id == MemoryEntry.id) | (MemoryRelation.target_id == MemoryEntry.id)
                ).filter(MemoryEntry.agent_id == agent_id)
            
            for relation_type, count in relation_query.all():
                relation_counts[relation_type] = count
            stats["relations"] = relation_counts
            
            # Get time range
            if total_count > 0:
                oldest = session.query(session.func.min(MemoryEntry.timestamp)).scalar()
                newest = session.query(session.func.max(MemoryEntry.timestamp)).scalar()
                
                stats["oldest_memory"] = oldest
                stats["newest_memory"] = newest
                stats["time_span_days"] = (newest - oldest) / (24 * 60 * 60) if oldest and newest else 0
            
            logger.debug(f"Retrieved memory database stats")
            return stats
        except Exception as e:
            logger.error(f"Error retrieving memory database stats: {str(e)}")
            return {"error": str(e)}
        finally:
            session.close()