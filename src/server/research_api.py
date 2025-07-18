"""
Research API for DeerFlow
Exposes research capabilities via REST API with mktagent integration
"""

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, List 
import json 
import logging

from src.research.task_manager import task_manager, ResearchStatus
from src.research.background_processor import BackgroundResearchProcessor

logger = logging.getLogger(__name__)

# Initialize background processor
background_processor = BackgroundResearchProcessor(task_manager)

router = APIRouter(
    prefix="/api/research", 
    tags=["Research"],
    responses={
        500: {"description": "Internal server error"},
        404: {"description": "Resource not found"},
        400: {"description": "Bad request"},
    }
)

class ResearchRequest(BaseModel):
    """
    Research request schema for submitting AI-powered research queries.
    
    This schema defines all parameters needed to configure the research workflow,
    from query planning to final report generation and summarization.
    """
    query: str = Field(
        ..., 
        min_length=3, 
        max_length=1000, 
        description="Research question or topic to investigate",
        example="Analyze the future outlook of mobile shooting games market including emerging trends, monetization strategies, and growth opportunities"
    )
    max_plan_iterations: int = Field(
        default=1, 
        ge=1, 
        le=5, 
        description="Maximum number of planning iterations for research strategy",
        example=2
    )
    max_step_num: int = Field(
        default=3, 
        ge=1, 
        le=10, 
        description="Maximum number of research steps to execute",
        example=5
    )
    enable_background_investigation: bool = Field(
        default=True, 
        description="Enable deep background investigation and additional source analysis",
        example=True
    )
    mktagent_campaign_id: Optional[int] = Field(
        default=None, 
        description="Optional MktAgent campaign ID for integration",
        example=12345
    )
    submit_to_mktagent: bool = Field(
        default=False, 
        description="Automatically submit completed research to MktAgent platform",
        example=False
    )

class ResearchResponse(BaseModel):
    """Response returned after successfully submitting a research request"""
    research_id: str = Field(description="Unique identifier for tracking the research task", example="research_12345_abcdef")
    status: str = Field(description="Initial status of the submitted task", example="pending")
    message: str = Field(description="Confirmation message", example="Research task submitted successfully")

class ResearchResult(BaseModel):
    """Comprehensive research result with AI-generated analysis, summary, and insights"""
    research_id: str = Field(description="Unique research task identifier", example="research_12345_abcdef")
    status: str = Field(description="Current processing status", example="completed")
    query: str = Field(description="Original research query", example="Mobile gaming market trends")
    created_at: str = Field(description="Task creation timestamp (ISO format)", example="2025-01-18T21:30:00Z")
    completed_at: Optional[str] = Field(default=None, description="Task completion timestamp (ISO format)", example="2025-01-18T21:35:00Z")
    final_report: Optional[str] = Field(default=None, description="Full research report with detailed analysis")
    summary: Optional[str] = Field(default=None, description="AI-generated concise summary (max 2 sentences)", example="The mobile shooting games market is expected to grow significantly from $8.5 billion in 2023 to $24-42 billion by 2032-33. Challenges include regulatory pressures and market saturation.")
    insights: Optional[List[str]] = Field(default=None, description="List of key insights extracted by AI", example=["Projected CAGR of 10-12.5% for mobile shooting games", "AR/VR integration driving innovation"])
    error: Optional[str] = Field(default=None, description="Error message if processing failed")
    mktagent_content_id: Optional[int] = Field(default=None, description="MktAgent content ID if submitted")
    processing_time: Optional[float] = Field(default=None, description="Total processing time in seconds", example=45.2)
    sources_analyzed: Optional[int] = Field(default=None, description="Number of sources analyzed during research", example=12)

class ResearchListResponse(BaseModel):
    """Research list response with pagination similar to mktagent"""
    data: List[ResearchResult]
    total: int
    page: int
    limit: int

@router.post("/submit", 
    response_model=ResearchResponse,
    summary="Submit Research Query",
    description="Submit a research query for AI-powered analysis and report generation with automatic summarization and insights extraction."
)
async def submit_research(
    request: ResearchRequest, 
    background_tasks: BackgroundTasks
):
    """
    Submit a research query for processing with the following workflow:
    
    1. **Deep Research**: AI agents perform comprehensive research
    2. **Report Generation**: Generate detailed research report
    3. **Summarization**: Extract concise summary and key insights
    4. **Background Processing**: All processing happens asynchronously
    
    **Parameters:**
    - `query`: Research question or topic (3-1000 characters)
    - `max_plan_iterations`: Planning iterations (1-5, default: 1)
    - `max_step_num`: Maximum research steps (1-10, default: 3)
    - `enable_background_investigation`: Enable deep research (default: True)
    - `submit_to_mktagent`: Auto-submit to MktAgent (default: False)
    
    **Returns:**
    - `research_id`: Unique identifier for tracking
    - `status`: Current processing status
    - `message`: Success confirmation
    
    **Example Usage:**
    ```json
    {
        "query": "Future trends in mobile gaming market 2025-2030",
        "max_plan_iterations": 2,
        "max_step_num": 5,
        "enable_background_investigation": true
    }
    ```
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

@router.get("/status/{research_id}", 
    response_model=ResearchResult,
    summary="Get Research Status",
    description="Retrieve the current status and results of a research task, including summary and insights when completed."
)
async def get_research_status(research_id: str):
    """
    Get the status and results of a research task.
    
    **Returns comprehensive information including:**
    - Task status (pending, running, completed, failed)
    - Final research report (when completed)
    - AI-generated summary (when completed)
    - Key insights list (when completed)
    - Processing metadata (timing, sources analyzed)
    - Error details (if failed)
    
    **Status Values:**
    - `pending`: Task queued for processing
    - `running`: Research in progress
    - `completed`: Research finished with results
    - `failed`: Processing encountered an error
    - `submitted_to_mktagent`: Results sent to MktAgent
    """
    try:
        task = task_manager.get_task(research_id)
        if not task:
            raise HTTPException(status_code=404, detail="Research task not found")
        
        # Parse insights from JSON string if available
        insights_list = []
        if task.insights:
            try:
                insights_list = json.loads(task.insights)
            except json.JSONDecodeError:
                insights_list = []
        
        return ResearchResult(
            research_id=task.id,
            status=task.status.value,
            query=task.query,
            created_at=task.created_at.isoformat(),
            completed_at=task.completed_at.isoformat() if task.completed_at else None,
            final_report=task.final_report,
            summary=task.summary,
            insights=insights_list,
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
        
        # Parse insights from JSON string if available
        insights_list = []
        if task.insights:
            try:
                insights_list = json.loads(task.insights)
            except json.JSONDecodeError:
                insights_list = []
        
        return {
            "research_id": task.id,
            "query": task.query,
            "final_report": task.final_report,
            "summary": task.summary,
            "insights": insights_list,
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
            # Parse insights from JSON string if available
            insights_list = []
            if task.insights:
                try:
                    insights_list = json.loads(task.insights)
                except json.JSONDecodeError:
                    insights_list = []
            
            task_results.append(ResearchResult(
                research_id=task.id,
                status=task.status.value,
                query=task.query,
                created_at=task.created_at.isoformat(),
                completed_at=task.completed_at.isoformat() if task.completed_at else None,
                final_report=task.final_report if task.status == ResearchStatus.COMPLETED else None,
                summary=task.summary,
                insights=insights_list,
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