"""
Research Task Manager for DeerFlow API
Manages research task lifecycle and status tracking
"""

import asyncio
import uuid
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ResearchStatus(Enum):
    """Research task status states"""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    SUBMITTED_TO_MKTAGENT = "submitted_to_mktagent"

@dataclass
class ResearchTask:
    """Research task data model"""
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

class ResearchTaskManager:
    """
    Manages research tasks similar to mktagent's task management patterns.
    Reuses patterns from mktagent's main.py for consistency.
    """
    
    def __init__(self):
        self.tasks: Dict[str, ResearchTask] = {}
        self._lock = asyncio.Lock()
    
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
        Similar to mktagent's content creation pattern.
        """
        task_id = f"research_{uuid.uuid4().hex[:8]}"
        
        task = ResearchTask(
            id=task_id,
            query=query,
            max_plan_iterations=max_plan_iterations,
            max_step_num=max_step_num,
            enable_background_investigation=enable_background_investigation,
            mktagent_campaign_id=mktagent_campaign_id,
            submit_to_mktagent=submit_to_mktagent
        )
        
        self.tasks[task_id] = task
        logger.info(f"Created research task {task_id} for query: {query[:50]}...")
        
        return task_id
    
    def get_task(self, task_id: str) -> Optional[ResearchTask]:
        """Get task by ID"""
        return self.tasks.get(task_id)
    
    async def update_task_status(
        self, 
        task_id: str, 
        status: ResearchStatus, 
        **kwargs
    ) -> bool:
        """
        Update task status and metadata.
        Thread-safe using async lock similar to mktagent patterns.
        """
        async with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                logger.error(f"Task {task_id} not found for status update")
                return False
            
            # Update status
            old_status = task.status
            task.status = status
            
            # Update additional fields
            for key, value in kwargs.items():
                if hasattr(task, key):
                    setattr(task, key, value)
            
            # Set completion time for terminal states
            if status in [ResearchStatus.COMPLETED, ResearchStatus.FAILED]:
                task.completed_at = datetime.now()
                if task.created_at:
                    task.processing_time = (task.completed_at - task.created_at).total_seconds()
            
            logger.info(f"Task {task_id} status updated: {old_status.value} -> {status.value}")
            return True
    
    def list_tasks(
        self, 
        status_filter: Optional[ResearchStatus] = None,
        limit: int = 100
    ) -> list[ResearchTask]:
        """
        List tasks with optional filtering.
        Similar to mktagent's pagination patterns.
        """
        tasks = list(self.tasks.values())
        
        if status_filter:
            tasks = [t for t in tasks if t.status == status_filter]
        
        # Sort by creation time, newest first
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        
        return tasks[:limit]
    
    def get_task_statistics(self) -> Dict[str, Any]:
        """Get task statistics for monitoring"""
        total_tasks = len(self.tasks)
        status_counts = {}
        
        for status in ResearchStatus:
            count = sum(1 for task in self.tasks.values() if task.status == status)
            status_counts[status.value] = count
        
        # Calculate average processing time for completed tasks
        completed_tasks = [t for t in self.tasks.values() 
                         if t.status == ResearchStatus.COMPLETED and t.processing_time]
        avg_processing_time = (
            sum(t.processing_time for t in completed_tasks) / len(completed_tasks)
            if completed_tasks else 0
        )
        
        return {
            "total_tasks": total_tasks,
            "status_counts": status_counts,
            "average_processing_time_seconds": round(avg_processing_time, 2),
            "completed_tasks": len(completed_tasks)
        }
    
    async def cleanup_old_tasks(self, max_age_hours: int = 24) -> int:
        """
        Clean up old completed/failed tasks.
        Similar to mktagent's cleanup patterns.
        """
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        removed_count = 0
        
        async with self._lock:
            tasks_to_remove = []
            
            for task_id, task in self.tasks.items():
                if (task.completed_at and 
                    task.completed_at < cutoff_time and 
                    task.status in [ResearchStatus.COMPLETED, ResearchStatus.FAILED]):
                    tasks_to_remove.append(task_id)
            
            for task_id in tasks_to_remove:
                del self.tasks[task_id]
                removed_count += 1
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old tasks")
        
        return removed_count

# Global task manager instance
# Similar to mktagent's global service instances pattern
task_manager = ResearchTaskManager()