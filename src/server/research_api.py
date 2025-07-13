"""
Research API for DeerFlow
Exposes research capabilities via REST API with mktagent integration
"""

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging

from src.research.task_manager import task_manager, ResearchStatus
from src.research.background_processor import BackgroundResearchProcessor

logger = logging.getLogger(__name__)

# Initialize background processor
background_processor = BackgroundResearchProcessor(task_manager)

router = APIRouter(prefix="/api/research", tags=["Research"])

class ResearchRequest(BaseModel):
    """
    Research request schema.
    Similar to mktagent's request patterns with validation.
    """
    query: str = Field(..., min_length=3, max_length=1000, description="Research query")
    max_plan_iterations: int = Field(default=1, ge=1, le=5, description="Maximum planning iterations")
    max_step_num: int = Field(default=3, ge=1, le=10, description="Maximum steps per plan")
    enable_background_investigation: bool = Field(default=True, description="Enable background research")
    mktagent_campaign_id: Optional[int] = Field(default=None, description="MktAgent campaign ID")
    submit_to_mktagent: bool = Field(default=False, description="Auto-submit results to mktagent")

class ResearchResponse(BaseModel):
    """Research submission response"""
    research_id: str
    status: str
    message: str

class ResearchResult(BaseModel):
    """Research result response"""
    research_id: str
    status: str
    query: str
    created_at: str
    completed_at: Optional[str] = None
    final_report: Optional[str] = None
    error: Optional[str] = None
    mktagent_content_id: Optional[int] = None
    processing_time: Optional[float] = None
    sources_analyzed: Optional[int] = None

class ResearchListResponse(BaseModel):
    """Research list response with pagination similar to mktagent"""
    data: List[ResearchResult]
    total: int
    page: int
    limit: int

@router.post("/submit", response_model=ResearchResponse)
async def submit_research(
    request: ResearchRequest, 
    background_tasks: BackgroundTasks
):
    """
    Submit a research query for processing.
    Similar to mktagent's task submission pattern from main.py:502-588
    """
    try:
        logger.info(f"Submitting research request: {request.query[:50]}...")
        
        # Create research task
        task_id = task_manager.create_task(
            query=request.query,
            max_plan_iterations=request.max_plan_iterations,
            max_step_num=request.max_step_num,
            enable_background_investigation=request.enable_background_investigation,
            mktagent_campaign_id=request.mktagent_campaign_id,
            submit_to_mktagent=request.submit_to_mktagent
        )
        
        # Start background processing
        # Following mktagent's background task pattern
        background_tasks.add_task(
            background_processor.submit_task_async,
            task_id
        )
        
        logger.info(f"Research task {task_id} submitted for background processing")
        
        return ResearchResponse(
            research_id=task_id,
            status="pending",
            message="Research task submitted successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to submit research request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{research_id}", response_model=ResearchResult)
async def get_research_status(research_id: str):
    """
    Get the status and results of a research task.
    Similar to mktagent's task status endpoint from main.py:591-647
    """
    try:
        task = task_manager.get_task(research_id)
        if not task:
            raise HTTPException(status_code=404, detail="Research task not found")
        
        return ResearchResult(
            research_id=task.id,
            status=task.status.value,
            query=task.query,
            created_at=task.created_at.isoformat(),
            completed_at=task.completed_at.isoformat() if task.completed_at else None,
            final_report=task.final_report,
            error=task.error,
            mktagent_content_id=task.mktagent_content_id,
            processing_time=task.processing_time,
            sources_analyzed=task.sources_analyzed
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get research status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/results/{research_id}")
async def get_research_results(research_id: str):
    """
    Get detailed research results including full report.
    Extended version of status endpoint for complete results.
    """
    try:
        task = task_manager.get_task(research_id)
        if not task:
            raise HTTPException(status_code=404, detail="Research task not found")
        
        if task.status != ResearchStatus.COMPLETED:
            raise HTTPException(
                status_code=400, 
                detail=f"Research not completed yet. Current status: {task.status.value}"
            )
        
        return {
            "research_id": task.id,
            "query": task.query,
            "final_report": task.final_report,
            "metadata": {
                "processing_time": task.processing_time,
                "sources_analyzed": task.sources_analyzed,
                "created_at": task.created_at.isoformat(),
                "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                "mktagent_content_id": task.mktagent_content_id,
                "submitted_to_mktagent": task.status == ResearchStatus.SUBMITTED_TO_MKTAGENT
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get research results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list", response_model=ResearchListResponse)
async def list_research_tasks(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(10, ge=1, le=100, description="Items per page"),
    status: Optional[str] = Query(None, description="Filter by status")
):
    """
    List research tasks with pagination.
    Similar to mktagent's list endpoints with pagination pattern.
    """
    try:
        # Parse status filter
        status_filter = None
        if status:
            try:
                status_filter = ResearchStatus(status)
            except ValueError:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid status. Valid values: {[s.value for s in ResearchStatus]}"
                )
        
        # Apply pagination offset
        offset = (page - 1) * limit
        
        # Get tasks with filtering and pagination from database
        paginated_tasks = task_manager.list_tasks(
            status_filter=status_filter,
            limit=limit,
            offset=offset
        )
        
        # Get total count for pagination
        # Note: This is a simple approach - for better performance, 
        # could optimize by adding count method to repository
        all_tasks = task_manager.list_tasks(status_filter=status_filter, limit=10000)
        total = len(all_tasks)
        
        # Convert to response format
        task_results = []
        for task in paginated_tasks:
            task_results.append(ResearchResult(
                research_id=task.id,
                status=task.status.value,
                query=task.query,
                created_at=task.created_at.isoformat(),
                completed_at=task.completed_at.isoformat() if task.completed_at else None,
                final_report=task.final_report if task.status == ResearchStatus.COMPLETED else None,
                error=task.error,
                mktagent_content_id=task.mktagent_content_id,
                processing_time=task.processing_time,
                sources_analyzed=task.sources_analyzed
            ))
        
        return ResearchListResponse(
            data=task_results,
            total=total,
            page=page,
            limit=limit
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list research tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/tasks/{research_id}")
async def delete_research_task(research_id: str):
    """
    Delete a research task.
    Similar to mktagent's delete patterns.
    """
    try:
        task = task_manager.get_task(research_id)
        if not task:
            raise HTTPException(status_code=404, detail="Research task not found")
        
        # Delete from database using task manager
        success = task_manager.delete_task(research_id)
        if success:
            logger.info(f"Deleted research task {research_id}")
            return {"message": "Research task deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Research task not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete research task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/statistics")
async def get_research_statistics():
    """
    Get research task statistics for monitoring.
    Useful for understanding system performance.
    """
    try:
        stats = task_manager.get_task_statistics()
        return stats
    except Exception as e:
        logger.error(f"Failed to get research statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))