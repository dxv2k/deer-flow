"""
Background Research Processor for DeerFlow
Handles async research execution and mktagent integration
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional
from datetime import datetime

from src.research.task_manager import ResearchTaskManager, ResearchStatus, task_manager
from src.workflow import run_agent_workflow_async
from src.integrations.mktagent_client import MktAgentClient

logger = logging.getLogger(__name__)

class BackgroundResearchProcessor:
    """
    Processes research tasks in background.
    Follows mktagent's background processing patterns from main.py
    """
    
    def __init__(self, task_manager: ResearchTaskManager, max_workers: int = 2):
        self.task_manager = task_manager
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.mktagent_client = None
        self._setup_mktagent_client()
    
    def _setup_mktagent_client(self):
        """Setup mktagent client if integration is enabled"""
        try:
            import os
            mktagent_url = os.getenv("MKTAGENT_API_URL", "http://localhost:8009")
            if mktagent_url:
                self.mktagent_client = MktAgentClient(mktagent_url)
                logger.info(f"MktAgent client configured: {mktagent_url}")
        except Exception as e:
            logger.warning(f"MktAgent client setup failed: {e}")
    
    async def process_research_task(
        self, 
        task_id: str
    ) -> Dict[str, Any]:
        """
        Process research task in background.
        Similar to mktagent's task processing pattern.
        """
        task = self.task_manager.get_task(task_id)
        if not task:
            logger.error(f"Task {task_id} not found")
            return {"error": "Task not found"}
        
        try:
            # Update status to running
            await self.task_manager.update_task_status(task_id, ResearchStatus.RUNNING)
            logger.info(f"Starting research for task {task_id}: {task.query}")
            
            # Execute research workflow
            start_time = datetime.now()
            result = await self._execute_research_workflow(task)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Extract results
            final_report = result.get("final_report", "")
            sources_count = len(result.get("observations", []))
            
            # Update task with results
            await self.task_manager.update_task_status(
                task_id,
                ResearchStatus.COMPLETED,
                final_report=final_report,
                processing_time=execution_time,
                sources_analyzed=sources_count
            )
            
            logger.info(f"Research completed for task {task_id} in {execution_time:.2f}s")
            
            # Submit to mktagent if requested
            if task.submit_to_mktagent and self.mktagent_client:
                await self._submit_to_mktagent(task_id, final_report)
            
            return {
                "task_id": task_id,
                "status": "completed",
                "final_report": final_report,
                "processing_time": execution_time,
                "sources_analyzed": sources_count
            }
            
        except Exception as e:
            error_msg = f"Research failed: {str(e)}"
            logger.error(f"Task {task_id} failed: {error_msg}")
            
            await self.task_manager.update_task_status(
                task_id,
                ResearchStatus.FAILED,
                error=error_msg
            )
            
            return {
                "task_id": task_id,
                "status": "failed",
                "error": error_msg
            }
    
    async def _execute_research_workflow(self, task) -> Dict[str, Any]:
        """
        Execute the DeerFlow research workflow.
        Reuses existing workflow with parameters from task.
        """
        try:
            result = await run_agent_workflow_async(
                user_input=task.query,
                debug=False,
                max_plan_iterations=task.max_plan_iterations,
                max_step_num=task.max_step_num,
                enable_background_investigation=task.enable_background_investigation
            )
            return result
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            raise
    
    async def _submit_to_mktagent(self, task_id: str, research_result: str):
        """
        Submit research results to mktagent.
        Uses mktagent's existing API patterns.
        """
        if not self.mktagent_client:
            logger.warning("MktAgent client not available")
            return
        
        task = self.task_manager.get_task(task_id)
        if not task:
            return
        
        try:
            logger.info(f"Submitting task {task_id} results to mktagent")
            
            # Submit as content to mktagent
            result = await self.mktagent_client.create_content(
                title=f"Research Report: {task.query}",
                content=research_result,
                campaign_id=task.mktagent_campaign_id
            )
            
            # Update task with mktagent content ID
            mktagent_content_id = result.get("id")
            if mktagent_content_id:
                await self.task_manager.update_task_status(
                    task_id,
                    ResearchStatus.SUBMITTED_TO_MKTAGENT,
                    mktagent_content_id=mktagent_content_id
                )
                logger.info(f"Task {task_id} submitted to mktagent as content {mktagent_content_id}")
            
        except Exception as e:
            logger.error(f"Failed to submit task {task_id} to mktagent: {e}")
            # Don't fail the whole task if mktagent submission fails
    
    async def submit_task_async(self, task_id: str):
        """
        Submit task for async processing.
        Entry point for FastAPI background tasks.
        """
        try:
            await self.process_research_task(task_id)
        except Exception as e:
            logger.error(f"Background task processing failed for {task_id}: {e}")

# Global processor instance
# Following mktagent's pattern of global service instances
background_processor = BackgroundResearchProcessor(task_manager)