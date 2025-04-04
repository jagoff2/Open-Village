"""
Virtual Computer System Module - Tools for agent interaction with computation

This module provides a secure, sandboxed environment where agents can execute
code, manipulate files, and utilize computational resources.

Dependencies:
- Python 3.10+
- pydantic 2.3+
- docker 6.1.2+
- langchain 0.0.267+
"""

import asyncio
import base64
import datetime
import docker
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable

import docker
from pydantic import BaseModel, Field

from agent_framework import Tool


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("virtual_computer.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class VirtualComputerConfig(BaseModel):
    """Configuration for a virtual computer"""
    name: str
    working_directory: str = "./virtual_computers"
    max_execution_time: int = 30  # seconds
    max_memory: int = 512  # MB
    enable_network: bool = False
    enable_file_access: bool = True
    allowed_languages: List[str] = ["python", "javascript", "bash"]
    custom_configuration: Dict[str, Any] = Field(default_factory=dict)


class VirtualComputer:
    """
    Virtual Computer for agent tool use
    
    This class provides a sandboxed environment for agents to execute code,
    manipulate files, and utilize computational resources.
    """
    def __init__(self, config: VirtualComputerConfig):
        self.id = str(uuid.uuid4())
        self.name = config.name
        
        # Set up working directory
        self.base_directory = os.path.join(config.working_directory, f"computer_{self.id}")
        os.makedirs(self.base_directory, exist_ok=True)
        
        # Initialize filesystem
        self.files_directory = os.path.join(self.base_directory, "files")
        os.makedirs(self.files_directory, exist_ok=True)
        
        # Initialize code directory
        self.code_directory = os.path.join(self.base_directory, "code")
        os.makedirs(self.code_directory, exist_ok=True)
        
        # Initialize output directory
        self.output_directory = os.path.join(self.base_directory, "output")
        os.makedirs(self.output_directory, exist_ok=True)
        
        # Configuration
        self.max_execution_time = config.max_execution_time
        self.max_memory = config.max_memory
        self.enable_network = config.enable_network
        self.enable_file_access = config.enable_file_access
        self.allowed_languages = set(config.allowed_languages)
        self.custom_config = config.custom_configuration
        
        # Docker client for sandboxed execution
        self.docker_client = docker.from_env()
        
        # Running processes
        self.running_processes: Dict[str, Dict[str, Any]] = {}
        
        # Available tools
        self.tools: Dict[str, Tool] = {}
        self._initialize_tools()
        
        logger.info(f"Initialized virtual computer {self.name} ({self.id})")
    
    def _initialize_tools(self) -> None:
        """Initialize the available tools"""
        # File system tools
        self.tools["list_files"] = Tool(
            name="list_files",
            description="List files in a directory",
            function=self.list_files,
            parameters={
                "path": {
                    "description": "Directory path to list files from",
                    "type": "string",
                    "default": "/"
                }
            }
        )
        
        self.tools["read_file"] = Tool(
            name="read_file",
            description="Read the contents of a file",
            function=self.read_file,
            parameters={
                "path": {
                    "description": "Path to the file to read",
                    "type": "string"
                }
            }
        )
        
        self.tools["write_file"] = Tool(
            name="write_file",
            description="Write content to a file",
            function=self.write_file,
            parameters={
                "path": {
                    "description": "Path to the file to write",
                    "type": "string"
                },
                "content": {
                    "description": "Content to write to the file",
                    "type": "string"
                }
            }
        )
        
        self.tools["delete_file"] = Tool(
            name="delete_file",
            description="Delete a file",
            function=self.delete_file,
            parameters={
                "path": {
                    "description": "Path to the file to delete",
                    "type": "string"
                }
            }
        )
        
        # Code execution tools
        self.tools["run_python"] = Tool(
            name="run_python",
            description="Execute Python code",
            function=self.run_python,
            parameters={
                "code": {
                    "description": "Python code to execute",
                    "type": "string"
                }
            }
        )
        
        self.tools["run_javascript"] = Tool(
            name="run_javascript",
            description="Execute JavaScript code",
            function=self.run_javascript,
            parameters={
                "code": {
                    "description": "JavaScript code to execute",
                    "type": "string"
                }
            }
        )
        
        self.tools["run_bash"] = Tool(
            name="run_bash",
            description="Execute Bash commands",
            function=self.run_bash,
            parameters={
                "commands": {
                    "description": "Bash commands to execute",
                    "type": "string"
                }
            }
        )
        
        # Project management tools
        self.tools["create_project"] = Tool(
            name="create_project",
            description="Create a new project directory with initial files",
            function=self.create_project,
            parameters={
                "project_name": {
                    "description": "Name of the project",
                    "type": "string"
                },
                "template": {
                    "description": "Template to use for the project",
                    "type": "string",
                    "enum": ["python", "web", "empty"],
                    "default": "empty"
                }
            }
        )
    
    def _sanitize_path(self, path: str) -> str:
        """
        Sanitize and resolve a file path to ensure it's within the allowed directory
        
        Args:
            path: The path to sanitize
            
        Returns:
            Sanitized absolute path
        """
        # Remove leading / if present
        if path.startswith("/"):
            path = path[1:]
        
        # Convert to absolute path within files directory
        absolute_path = os.path.abspath(os.path.join(self.files_directory, path))
        
        # Ensure path is within files directory
        if not absolute_path.startswith(self.files_directory):
            logger.warning(f"Attempted to access path outside files directory: {path}")
            raise ValueError(f"Access denied: {path} is outside the allowed directory")
        
        return absolute_path
    
    def list_files(self, path: str = "/") -> Dict[str, Any]:
        """
        List files in a directory
        
        Args:
            path: Directory path to list files from
            
        Returns:
            Dictionary containing file listing
        """
        try:
            # Sanitize path
            absolute_path = self._sanitize_path(path)
            
            # Check if directory exists
            if not os.path.exists(absolute_path):
                return {"error": f"Directory {path} does not exist"}
            
            # Check if path is a directory
            if not os.path.isdir(absolute_path):
                return {"error": f"{path} is not a directory"}
            
            # List directory contents
            contents = os.listdir(absolute_path)
            
            # Get details for each item
            items = []
            for item in contents:
                item_path = os.path.join(absolute_path, item)
                item_stat = os.stat(item_path)
                
                # Convert path to relative for the result
                relative_path = os.path.relpath(item_path, self.files_directory)
                if relative_path == ".":
                    relative_path = "/"
                else:
                    relative_path = "/" + relative_path
                
                items.append({
                    "name": item,
                    "path": relative_path,
                    "type": "directory" if os.path.isdir(item_path) else "file",
                    "size": item_stat.st_size,
                    "modified": datetime.datetime.fromtimestamp(item_stat.st_mtime).isoformat()
                })
            
            return {
                "path": path,
                "items": items
            }
        except Exception as e:
            logger.error(f"Error listing files in {path}: {str(e)}")
            return {"error": str(e)}
    
    def read_file(self, path: str) -> Dict[str, Any]:
        """
        Read the contents of a file
        
        Args:
            path: Path to the file to read
            
        Returns:
            Dictionary containing file contents
        """
        try:
            # Sanitize path
            absolute_path = self._sanitize_path(path)
            
            # Check if file exists
            if not os.path.exists(absolute_path):
                return {"error": f"File {path} does not exist"}
            
            # Check if path is a file
            if not os.path.isfile(absolute_path):
                return {"error": f"{path} is not a file"}
            
            # Read file contents
            with open(absolute_path, "r") as f:
                content = f.read()
            
            # Get file info
            file_stat = os.stat(absolute_path)
            
            return {
                "path": path,
                "content": content,
                "size": file_stat.st_size,
                "modified": datetime.datetime.fromtimestamp(file_stat.st_mtime).isoformat()
            }
        except Exception as e:
            logger.error(f"Error reading file {path}: {str(e)}")
            return {"error": str(e)}
    
    def write_file(self, path: str, content: str) -> Dict[str, Any]:
        """
        Write content to a file
        
        Args:
            path: Path to the file to write
            content: Content to write to the file
            
        Returns:
            Dictionary containing result of operation
        """
        try:
            # Sanitize path
            absolute_path = self._sanitize_path(path)
            
            # Create parent directories if they don't exist
            os.makedirs(os.path.dirname(absolute_path), exist_ok=True)
            
            # Write file contents
            with open(absolute_path, "w") as f:
                f.write(content)
            
            # Get file info
            file_stat = os.stat(absolute_path)
            
            return {
                "path": path,
                "size": file_stat.st_size,
                "modified": datetime.datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                "success": True
            }
        except Exception as e:
            logger.error(f"Error writing to file {path}: {str(e)}")
            return {"error": str(e), "success": False}
    
    def delete_file(self, path: str) -> Dict[str, Any]:
        """
        Delete a file
        
        Args:
            path: Path to the file to delete
            
        Returns:
            Dictionary containing result of operation
        """
        try:
            # Sanitize path
            absolute_path = self._sanitize_path(path)
            
            # Check if file exists
            if not os.path.exists(absolute_path):
                return {"error": f"File {path} does not exist", "success": False}
            
            # Delete file or directory
            if os.path.isdir(absolute_path):
                shutil.rmtree(absolute_path)
            else:
                os.remove(absolute_path)
            
            return {
                "path": path,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error deleting file {path}: {str(e)}")
            return {"error": str(e), "success": False}
    
    def run_python(self, code: str) -> Dict[str, Any]:
        """
        Execute Python code in a sandboxed environment
        
        Args:
            code: Python code to execute
            
        Returns:
            Dictionary containing execution result
        """
        if "python" not in self.allowed_languages:
            return {"error": "Python execution is not allowed", "success": False}
        
        try:
            # Generate a unique ID for this execution
            execution_id = str(uuid.uuid4())
            
            # Create a temporary file for the code
            code_file_path = os.path.join(self.code_directory, f"{execution_id}.py")
            with open(code_file_path, "w") as f:
                f.write(code)
            
            # Create output file
            output_file_path = os.path.join(self.output_directory, f"{execution_id}.out")
            
            # Run the code in a Docker container
            container = self.docker_client.containers.run(
                "python:3.10-slim",
                command=f"python /code/{execution_id}.py",
                volumes={
                    self.code_directory: {"bind": "/code", "mode": "ro"},
                    self.files_directory: {"bind": "/files", "mode": "rw"},
                    self.output_directory: {"bind": "/output", "mode": "rw"}
                },
                environment={
                    "PYTHONUNBUFFERED": "1"
                },
                network_mode="none" if not self.enable_network else "bridge",
                mem_limit=f"{self.max_memory}m",
                detach=True,
                remove=True
            )
            
            # Track the running process
            start_time = time.time()
            self.running_processes[execution_id] = {
                "container_id": container.id,
                "start_time": start_time,
                "language": "python",
                "code_file": code_file_path,
                "output_file": output_file_path
            }
            
            # Wait for completion or timeout
            try:
                exit_code = container.wait(timeout=self.max_execution_time)["StatusCode"]
            except Exception as e:
                logger.warning(f"Execution timed out or failed: {str(e)}")
                try:
                    container.stop()
                except:
                    pass
                return {"error": f"Execution timed out or failed: {str(e)}", "success": False}
            

            # Check for output file
            output = ""
            if os.path.exists(output_file_path):
                with open(output_file_path, "r") as f:
                    output = f.read()
            
            # Get stdout and stderr from container logs
            logs = container.logs().decode("utf-8")
            
            # Clean up
            del self.running_processes[execution_id]
            
            return {
                "execution_id": execution_id,
                "exit_code": exit_code,
                "output": output,
                "logs": logs,
                "execution_time": time.time() - start_time,
                "success": exit_code == 0
            }
        except Exception as e:
            logger.error(f"Error executing Python code: {str(e)}")
            return {"error": str(e), "success": False}
    
    def run_javascript(self, code: str) -> Dict[str, Any]:
        """
        Execute JavaScript code in a sandboxed environment
        
        Args:
            code: JavaScript code to execute
            
        Returns:
            Dictionary containing execution result
        """
        if "javascript" not in self.allowed_languages:
            return {"error": "JavaScript execution is not allowed", "success": False}
        
        try:
            # Generate a unique ID for this execution
            execution_id = str(uuid.uuid4())
            
            # Create a temporary file for the code
            code_file_path = os.path.join(self.code_directory, f"{execution_id}.js")
            with open(code_file_path, "w") as f:
                f.write(code)
            
            # Create output file
            output_file_path = os.path.join(self.output_directory, f"{execution_id}.out")
            
            # Run the code in a Docker container
            container = self.docker_client.containers.run(
                "node:16-alpine",
                command=f"node /code/{execution_id}.js",
                volumes={
                    self.code_directory: {"bind": "/code", "mode": "ro"},
                    self.files_directory: {"bind": "/files", "mode": "rw"},
                    self.output_directory: {"bind": "/output", "mode": "rw"}
                },
                network_mode="none" if not self.enable_network else "bridge",
                mem_limit=f"{self.max_memory}m",
                detach=True,
                remove=True
            )
            
            # Track the running process
            start_time = time.time()
            self.running_processes[execution_id] = {
                "container_id": container.id,
                "start_time": start_time,
                "language": "javascript",
                "code_file": code_file_path,
                "output_file": output_file_path
            }
            
            # Wait for completion or timeout
            try:
                exit_code = container.wait(timeout=self.max_execution_time)["StatusCode"]
            except Exception as e:
                logger.warning(f"Execution timed out or failed: {str(e)}")
                try:
                    container.stop()
                except:
                    pass
                return {"error": f"Execution timed out or failed: {str(e)}", "success": False}
            
            # Check for output file
            output = ""
            if os.path.exists(output_file_path):
                with open(output_file_path, "r") as f:
                    output = f.read()
            
            # Get stdout and stderr from container logs
            logs = container.logs().decode("utf-8")
            
            # Clean up
            del self.running_processes[execution_id]
            
            return {
                "execution_id": execution_id,
                "exit_code": exit_code,
                "output": output,
                "logs": logs,
                "execution_time": time.time() - start_time,
                "success": exit_code == 0
            }
        except Exception as e:
            logger.error(f"Error executing JavaScript code: {str(e)}")
            return {"error": str(e), "success": False}
    
    def run_bash(self, commands: str) -> Dict[str, Any]:
        """
        Execute Bash commands in a sandboxed environment
        
        Args:
            commands: Bash commands to execute
            
        Returns:
            Dictionary containing execution result
        """
        if "bash" not in self.allowed_languages:
            return {"error": "Bash execution is not allowed", "success": False}
        
        try:
            # Generate a unique ID for this execution
            execution_id = str(uuid.uuid4())
            
            # Create a temporary file for the commands
            command_file_path = os.path.join(self.code_directory, f"{execution_id}.sh")
            with open(command_file_path, "w") as f:
                f.write("#!/bin/bash\n")
                f.write(commands)
            
            # Create output file
            output_file_path = os.path.join(self.output_directory, f"{execution_id}.out")
            
            # Run the commands in a Docker container
            container = self.docker_client.containers.run(
                "alpine:latest",
                command=f"sh /code/{execution_id}.sh",
                volumes={
                    self.code_directory: {"bind": "/code", "mode": "ro"},
                    self.files_directory: {"bind": "/files", "mode": "rw"},
                    self.output_directory: {"bind": "/output", "mode": "rw"}
                },
                network_mode="none" if not self.enable_network else "bridge",
                mem_limit=f"{self.max_memory}m",
                detach=True,
                remove=True
            )
            
            # Track the running process
            start_time = time.time()
            self.running_processes[execution_id] = {
                "container_id": container.id,
                "start_time": start_time,
                "language": "bash",
                "code_file": command_file_path,
                "output_file": output_file_path
            }
            
            # Wait for completion or timeout
            try:
                exit_code = container.wait(timeout=self.max_execution_time)["StatusCode"]
            except Exception as e:
                logger.warning(f"Execution timed out or failed: {str(e)}")
                try:
                    container.stop()
                except:
                    pass
                return {"error": f"Execution timed out or failed: {str(e)}", "success": False}
            
            # Check for output file
            output = ""
            if os.path.exists(output_file_path):
                with open(output_file_path, "r") as f:
                    output = f.read()
            
            # Get stdout and stderr from container logs
            logs = container.logs().decode("utf-8")
            
            # Clean up
            del self.running_processes[execution_id]
            
            return {
                "execution_id": execution_id,
                "exit_code": exit_code,
                "output": output,
                "logs": logs,
                "execution_time": time.time() - start_time,
                "success": exit_code == 0
            }
        except Exception as e:
            logger.error(f"Error executing Bash commands: {str(e)}")
            return {"error": str(e), "success": False}
    
    def create_project(self, project_name: str, template: str = "empty") -> Dict[str, Any]:
        """
        Create a new project directory with initial files
        
        Args:
            project_name: Name of the project
            template: Template to use for the project
            
        Returns:
            Dictionary containing result of operation
        """
        try:
            # Sanitize project name - only allow alphanumeric and underscore
            if not re.match(r'^[a-zA-Z0-9_]+$', project_name):
                return {"error": "Project name must contain only letters, numbers, and underscores", "success": False}
            
            # Create project directory
            project_path = self._sanitize_path(project_name)
            
            if os.path.exists(project_path):
                return {"error": f"Project {project_name} already exists", "success": False}
            
            os.makedirs(project_path)
            
            # Initialize project based on template
            if template == "python":
                # Create basic Python project structure
                os.makedirs(os.path.join(project_path, "src"))
                os.makedirs(os.path.join(project_path, "tests"))
                os.makedirs(os.path.join(project_path, "docs"))
                
                # Create initial files
                with open(os.path.join(project_path, "README.md"), "w") as f:
                    f.write(f"# {project_name}\n\nA Python project.\n")
                
                with open(os.path.join(project_path, "src", "__init__.py"), "w") as f:
                    f.write("")
                
                with open(os.path.join(project_path, "src", "main.py"), "w") as f:
                    f.write(f"""
def main():
    print("Hello from {project_name}!")

if __name__ == "__main__":
    main()
""")
                
                with open(os.path.join(project_path, "tests", "__init__.py"), "w") as f:
                    f.write("")
                
                with open(os.path.join(project_path, "tests", "test_main.py"), "w") as f:
                    f.write(f"""
from src.main import main

def test_main():
    # TODO: Write tests
    pass
""")
                
                with open(os.path.join(project_path, "requirements.txt"), "w") as f:
                    f.write("# Dependencies\n")
            
            elif template == "web":
                # Create basic web project structure
                os.makedirs(os.path.join(project_path, "css"))
                os.makedirs(os.path.join(project_path, "js"))
                os.makedirs(os.path.join(project_path, "img"))
                
                # Create initial files
                with open(os.path.join(project_path, "index.html"), "w") as f:
                    f.write(f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{project_name}</title>
    <link rel="stylesheet" href="css/styles.css">
</head>
<body>
    <h1>Welcome to {project_name}</h1>
    <script src="js/main.js"></script>
</body>
</html>
""")
                
                with open(os.path.join(project_path, "css", "styles.css"), "w") as f:
                    f.write("""body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 20px;
}

h1 {
    color: #333;
}
""")
                
                with open(os.path.join(project_path, "js", "main.js"), "w") as f:
                    f.write("""// Main JavaScript file
console.log('Script loaded!');
""")
                
                with open(os.path.join(project_path, "README.md"), "w") as f:
                    f.write(f"# {project_name}\n\nA web project.\n")
            
            else:  # empty template
                # Create just a basic README
                with open(os.path.join(project_path, "README.md"), "w") as f:
                    f.write(f"# {project_name}\n\nNew project.\n")
            
            return {
                "project_name": project_name,
                "path": f"/{project_name}",
                "template": template,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error creating project {project_name}: {str(e)}")
            return {"error": str(e), "success": False}
    
    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """
        Get a tool by name
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            The tool if it exists, None otherwise
        """
        return self.tools.get(tool_name)
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """
        Get information about available tools
        
        Returns:
            List of dictionaries containing tool information
        """
        return [
            {
                "name": name,
                "description": tool.description,
                "parameters": tool.parameters
            }
            for name, tool in self.tools.items()
        ]
    
    def cleanup(self) -> None:
        """
        Clean up resources and stop any running processes
        """
        # Stop all running processes
        for execution_id, process_info in list(self.running_processes.items()):
            try:
                container_id = process_info.get("container_id")
                if container_id:
                    container = self.docker_client.containers.get(container_id)
                    container.stop()
            except Exception as e:
                logger.error(f"Error stopping container for execution {execution_id}: {str(e)}")
        
        # Clear running processes
        self.running_processes.clear()
        
        logger.info(f"Cleaned up resources for virtual computer {self.name}")