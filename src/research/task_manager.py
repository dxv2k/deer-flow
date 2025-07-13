"""
Research Task Manager for DeerFlow API
Manages research task lifecycle and status tracking with persistent storage
"""

import asyncio
import uuid
from typing import Dict, Optional, Any, List
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import logging

from src.research.database import (
    ResearchTaskRepository, 
    ResearchTaskDB, 
    ResearchStatusEnum,
    init_database
)

logger = logging.getLogger(__name__)

# Re-export for backward compatibility
ResearchStatus = ResearchStatusEnum

@dataclass
class ResearchTask:
    """
    In-memory research task data model.
    Now backed by persistent database storage.
    """
    id: str
    query: str
    status: ResearchStatus = ResearchStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    final_report: Optional[str] = None
    error: Optional[str] = None
    mktagent_content_id: Optional[int] = None
    processing_time: Optional[float] = None
    sources_analyzed: Optional[int] = None
    
    # Configuration parameters
    max_plan_iterations: int = 1
    max_step_num: int = 3
    enable_background_investigation: bool = True
    mktagent_campaign_id: Optional[int] = None
    submit_to_mktagent: bool = False
    
    @classmethod
    def from_db(cls, db_task: ResearchTaskDB) -> 'ResearchTask':
        """Create ResearchTask from database model"""
        return cls(
            id=db_task.id,
            query=db_task.query,
            status=db_task.status,
            created_at=db_task.created_at,
            completed_at=db_task.completed_at,
            final_report=db_task.final_report,
            error=db_task.error,
            mktagent_content_id=db_task.mktagent_content_id,
            processing_time=db_task.processing_time,
            sources_analyzed=db_task.sources_analyzed,
            max_plan_iterations=db_task.max_plan_iterations,
            max_step_num=db_task.max_step_num,
            enable_background_investigation=db_task.enable_background_investigation,
            mktagent_campaign_id=db_task.mktagent_campaign_id,
            submit_to_mktagent=db_task.submit_to_mktagent
        )

class ResearchTaskManager:
    """
    Manages research tasks with persistent SQLite storage.
    Now uses database backend instead of in-memory storage.
    """
    
    def __init__(self):
        self.repository = ResearchTaskRepository()
        self._lock = asyncio.Lock()
        
        # Initialize database on startup
        try:
            init_database()
            logger.info("ResearchTaskManager initialized with persistent storage")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def create_task(
        self, 
        query: str,
        max_plan_iterations: int = 1,
        max_step_num: int = 3,
        enable_background_investigation: bool = True,
        mktagent_campaign_id: Optional[int] = None,
        submit_to_mktagent: bool = False
    ) -> str:
        """
        Create a new research task and return its ID.
        Now persists to database.
        """
        task_id = f"research_{uuid.uuid4().hex[:8]}"
        
        try:
            # Create task in database
            self.repository.create_task(
                task_id=task_id,
                query=query,
                max_plan_iterations=max_plan_iterations,
                max_step_num=max_step_num,
                enable_background_investigation=enable_background_investigation,
                mktagent_campaign_id=mktagent_campaign_id,
                submit_to_mktagent=submit_to_mktagent
            )
            
            logger.info(f"Created persistent research task {task_id} for query: {query[:50]}...")
            return task_id
            
        except Exception as e:
            logger.error(f"Failed to create task {task_id}: {e}")
            raise
    
    def get_task(self, task_id: str) -> Optional[ResearchTask]:
        """Get task by ID from database"""
        try:
            db_task = self.repository.get_task(task_id)
            if db_task:
                return ResearchTask.from_db(db_task)
            return None
        except Exception as e:
            logger.error(f"Failed to get task {task_id}: {e}")
            return None
    
    async def update_task_status(
        self, 
        task_id: str, 
        status: ResearchStatus, 
        **kwargs
    ) -> bool:
        """
        Update task status and metadata in database.
        Thread-safe using async lock.
        """
        async with self._lock:
            try:
                success = self.repository.update_task_status(
                    task_id=task_id,
                    status=status,
                    **kwargs
                )
                return success
            except Exception as e:
                logger.error(f"Failed to update task {task_id}: {e}")
                return False
    
    def list_tasks(
        self, 
        status_filter: Optional[ResearchStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[ResearchTask]:
        """
        List tasks with optional filtering from database.
        """
        try:
            db_tasks, _ = self.repository.list_tasks(
                status_filter=status_filter,
                limit=limit,
                offset=offset
            )
            
            # Convert to in-memory objects
            tasks = [ResearchTask.from_db(db_task) for db_task in db_tasks]
            return tasks
            
        except Exception as e:
            logger.error(f"Failed to list tasks: {e}")
            return []
    
    def get_task_statistics(self) -> Dict[str, Any]:
        """Get task statistics from database"""
        try:
            return self.repository.get_statistics()
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
    
    async def cleanup_old_tasks(self, max_age_hours: int = 24) -> int:
        """
        Clean up old completed/failed tasks from database.
        """
        async with self._lock:
            try:
                removed_count = self.repository.cleanup_old_tasks(max_age_hours)
                return removed_count
            except Exception as e:
                logger.error(f"Failed to cleanup old tasks: {e}")
                return 0
    
    def delete_task(self, task_id: str) -> bool:
        """Delete a task from database"""
        try:
            return self.repository.delete_task(task_id)
        except Exception as e:
            logger.error(f"Failed to delete task {task_id}: {e}")
            return False

# Global task manager instance
# Similar to mktagent's global service instances pattern
task_manager = ResearchTaskManager()