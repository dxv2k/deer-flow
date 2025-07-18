"""
Background Research Processor for DeerFlow
Handles async research execution and mktagent integration
"""

import asyncio
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional
from datetime import datetime

from src.research.task_manager import ResearchTaskManager, ResearchStatus, task_manager
from src.graph import build_graph
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
            error_msg = f"Task {task_id} not found"
            logger.error(error_msg)
            return {"error": error_msg}
        
        try:
            # Update status to running
            await self.task_manager.update_task_status(task_id, ResearchStatus.RUNNING)
            logger.info(f"Starting research for task {task_id}: {task.query}")
            
            # Execute research workflow
            start_time = datetime.now()
            try:
                result = await self._execute_research_workflow(task)
                logger.info(f"Workflow execution completed for task {task_id}")
            except Exception as workflow_error:
                error_msg = f"Workflow execution failed: {str(workflow_error)}"
                logger.error(f"Task {task_id} workflow error: {error_msg}")
                logger.error(f"Full traceback: {traceback.format_exc()}")
                
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
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Extract results with comprehensive error handling
            try:
                if result is None:
                    raise ValueError("Workflow returned None result")
                
                if not isinstance(result, dict):
                    raise ValueError(f"Workflow returned unexpected type: {type(result)}, expected dict")
                
                final_report = result.get("final_report", "")
                if not final_report:
                    logger.warning(f"Task {task_id}: No final_report in result. Available keys: {list(result.keys())}")
                    final_report = "No report generated"
                
                observations = result.get("observations", [])
                sources_count = len(observations) if observations else 0
                
                logger.info(f"Task {task_id}: Extracted final_report length: {len(final_report)}, sources: {sources_count}")
                
            except Exception as extract_error:
                error_msg = f"Result extraction failed: {str(extract_error)}"
                logger.error(f"Task {task_id} extraction error: {error_msg}")
                logger.error(f"Result object: {result}")
                logger.error(f"Full traceback: {traceback.format_exc()}")
                
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
            
            # Generate summary and insights after successful research
            summary = None
            insights = []
            try:
                from src.research.summarization import summarize_research_result
                
                logger.info(f"Task {task_id}: Generating summary and insights from final report")
                summarization_result = summarize_research_result(final_report)
                
                if summarization_result.get("success", False):
                    summary = summarization_result.get("summary")
                    insights = summarization_result.get("insights", [])
                    logger.info(f"Task {task_id}: Successfully generated summary ({len(summary) if summary else 0} chars) and {len(insights)} insights")
                else:
                    error_msg = summarization_result.get("error", "Unknown summarization error")
                    logger.warning(f"Task {task_id}: Summarization failed but continuing: {error_msg}")
                    
            except Exception as summarization_error:
                logger.warning(f"Task {task_id}: Summarization failed but continuing research completion: {str(summarization_error)}")
                logger.warning(f"Summarization traceback: {traceback.format_exc()}")
            
            # Update task with results including summary and insights
            try:
                update_success = await self.task_manager.update_task_status(
                    task_id,
                    ResearchStatus.COMPLETED,
                    final_report=final_report,
                    summary=summary,
                    insights=insights,
                    processing_time=execution_time,
                    sources_analyzed=sources_count
                )
                
                if not update_success:
                    logger.error(f"Failed to update task {task_id} status to completed")
                
            except Exception as update_error:
                error_msg = f"Status update failed: {str(update_error)}"
                logger.error(f"Task {task_id} update error: {error_msg}")
                logger.error(f"Full traceback: {traceback.format_exc()}")
                
                return {
                    "task_id": task_id,
                    "status": "failed",
                    "error": error_msg
                }
            
            logger.info(f"Research completed for task {task_id} in {execution_time:.2f}s")
            
            # Submit to mktagent if requested
            if task.submit_to_mktagent and self.mktagent_client:
                try:
                    await self._submit_to_mktagent(task_id, final_report)
                except Exception as mktagent_error:
                    logger.error(f"MktAgent submission failed for task {task_id}: {str(mktagent_error)}")
                    logger.error(f"Full traceback: {traceback.format_exc()}")
                    # Don't fail the whole task if mktagent submission fails
            
            return {
                "task_id": task_id,
                "status": "completed",
                "final_report": final_report,
                "summary": summary,
                "insights": insights,
                "processing_time": execution_time,
                "sources_analyzed": sources_count
            }
            
        except Exception as e:
            error_msg = f"Research processing failed: {str(e)}"
            logger.error(f"Task {task_id} unexpected error: {error_msg}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            try:
                await self.task_manager.update_task_status(
                    task_id,
                    ResearchStatus.FAILED,
                    error=error_msg
                )
            except Exception as update_error:
                logger.error(f"Failed to update task {task_id} status after error: {str(update_error)}")
            
            return {
                "task_id": task_id,
                "status": "failed",
                "error": error_msg
            }
    
    async def _execute_research_workflow(self, task) -> Dict[str, Any]:
        """
        Execute the DeerFlow research workflow.
        Modified to properly capture final state instead of using the console version.
        """
        try:
            logger.info(f"Executing workflow for query: {task.query}")
            
            # Build the graph directly for API usage
            graph = build_graph()
            
            # Create initial state similar to the server app pattern
            initial_state = {
                "messages": [{"role": "user", "content": task.query}],
                "plan_iterations": 0,
                "final_report": "",
                "current_plan": None,
                "observations": [],
                "auto_accepted_plan": True,  # Skip human feedback for background processing
                "enable_background_investigation": task.enable_background_investigation,
                "research_topic": task.query,
            }
            
            # Configuration for the workflow
            config = {
                "configurable": {
                    "thread_id": f"background_{task.id}",
                    "max_plan_iterations": task.max_plan_iterations,
                    "max_step_num": task.max_step_num,
                    "mcp_settings": {
                        "servers": {
                            "mcp-github-trending": {
                                "transport": "stdio",
                                "command": "uvx",
                                "args": ["mcp-github-trending"],
                                "enabled_tools": ["get_github_trending_repositories"],
                                "add_to_agents": ["researcher"],
                            }
                        }
                    },
                },
                "recursion_limit": 100,
            }
            
            logger.info(f"Starting workflow execution with config: {config['configurable']}")
            
            # Execute the workflow and capture the final state
            final_state = None
            async for state in graph.astream(
                input=initial_state, 
                config=config, 
                stream_mode="values"
            ):
                final_state = state
                logger.debug(f"Workflow state update: {list(state.keys()) if isinstance(state, dict) else type(state)}")
            
            if final_state is None:
                raise ValueError("Workflow completed without returning final state")
            
            logger.info(f"Workflow completed. Final state keys: {list(final_state.keys()) if isinstance(final_state, dict) else 'Not a dict'}")
            
            # Validate the final state
            if not isinstance(final_state, dict):
                raise ValueError(f"Final state is not a dictionary: {type(final_state)}")
            
            # Log what we got for debugging
            final_report = final_state.get("final_report", "")
            observations = final_state.get("observations", [])
            logger.info(f"Final report length: {len(final_report) if final_report else 0}")
            logger.info(f"Observations count: {len(observations) if observations else 0}")
            
            return final_state
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
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
            logger.warning(f"Task {task_id} not found for mktagent submission")
            return
        
        try:
            logger.info(f"Submitting task {task_id} results to mktagent")
            
            # Validate research result
            if not research_result or not research_result.strip():
                raise ValueError("Research result is empty or whitespace only")
            
            # Submit as content to mktagent
            result = await self.mktagent_client.create_content(
                title=f"Research Report: {task.query}",
                content=research_result,
                campaign_id=task.mktagent_campaign_id
            )
            
            if not result:
                raise ValueError("MktAgent returned empty result")
            
            # Update task with mktagent content ID
            mktagent_content_id = result.get("id")
            if mktagent_content_id:
                try:
                    update_success = await self.task_manager.update_task_status(
                        task_id,
                        ResearchStatus.SUBMITTED_TO_MKTAGENT,
                        mktagent_content_id=mktagent_content_id
                    )
                    if update_success:
                        logger.info(f"Task {task_id} submitted to mktagent as content {mktagent_content_id}")
                    else:
                        logger.error(f"Failed to update task {task_id} with mktagent content ID")
                except Exception as update_error:
                    logger.error(f"Failed to update task {task_id} status after mktagent submission: {str(update_error)}")
                    logger.error(f"Full traceback: {traceback.format_exc()}")
            else:
                logger.warning(f"MktAgent response missing 'id' field: {result}")
            
        except Exception as e:
            logger.error(f"Failed to submit task {task_id} to mktagent: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            # Don't fail the whole task if mktagent submission fails
    
    async def submit_task_async(self, task_id: str):
        """
        Submit task for async processing.
        Entry point for FastAPI background tasks.
        """
        try:
            result = await self.process_research_task(task_id)
            logger.info(f"Background processing completed for task {task_id}: {result.get('status', 'unknown')}")
        except Exception as e:
            logger.error(f"Background task processing failed for {task_id}: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            # Try to update task status to failed if possible
            try:
                await self.task_manager.update_task_status(
                    task_id,
                    ResearchStatus.FAILED,
                    error=f"Background processing error: {str(e)}"
                )
            except Exception as update_error:
                logger.error(f"Failed to update task {task_id} status after background error: {str(update_error)}")

# Global processor instance
# Following mktagent's pattern of global service instances
background_processor = BackgroundResearchProcessor(task_manager)