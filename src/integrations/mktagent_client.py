"""
MktAgent API Client for DeerFlow
Handles communication with mktagent's existing APIs
"""

import httpx
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class MktAgentClient:
    """
    HTTP client for mktagent API integration.
    Uses mktagent's existing endpoints without requiring changes.
    """
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.timeout = 30.0
        
    async def create_content(
        self, 
        title: str,
        content: str, 
        campaign_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Create content in mktagent via existing /contents/ API.
        Reuses mktagent's content creation pattern from main.py:76-94
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            content_data = {
                "title": title,
                "content": content,
                "status": "COMPLETED"  # Mark as completed since research is done
            }
            
            # Add campaign_id if provided
            if campaign_id:
                content_data["campaign_id"] = campaign_id
            
            try:
                response = await client.post(
                    f"{self.base_url}/contents/",
                    json=content_data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"Successfully created content in mktagent: {result.get('id')}")
                    return result
                else:
                    error_msg = f"MktAgent API error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    raise Exception(error_msg)
                    
            except httpx.RequestError as e:
                error_msg = f"Request to mktagent failed: {str(e)}"
                logger.error(error_msg)
                raise Exception(error_msg)
    
    async def get_campaign_info(self, campaign_id: int) -> Dict[str, Any]:
        """
        Get campaign information from mktagent.
        Uses mktagent's existing /campaigns/{id} endpoint from main.py:302-328
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(
                    f"{self.base_url}/campaigns/{campaign_id}"
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"Retrieved campaign {campaign_id} info from mktagent")
                    return result
                elif response.status_code == 404:
                    raise Exception(f"Campaign {campaign_id} not found in mktagent")
                else:
                    error_msg = f"MktAgent API error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    raise Exception(error_msg)
                    
            except httpx.RequestError as e:
                error_msg = f"Request to mktagent failed: {str(e)}"
                logger.error(error_msg)
                raise Exception(error_msg)
    
    async def trigger_content_generation(
        self,
        research_content: str,
        campaign_id: int,
        text_prompt: str
    ) -> Dict[str, Any]:
        """
        Trigger content generation based on research results.
        Uses mktagent's existing /tasks/generate-content endpoint from main.py:502-588
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            generation_request = {
                "text_prompt": text_prompt,
                "campaign_id": campaign_id,
                "image_prompt": f"Visual representation related to: {research_content[:200]}...",
                "platform": "instagram"  # Default platform
            }
            
            try:
                response = await client.post(
                    f"{self.base_url}/tasks/generate-content",
                    json=generation_request
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"Successfully triggered content generation: task {result.get('task_id')}")
                    return result
                else:
                    error_msg = f"Content generation API error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    raise Exception(error_msg)
                    
            except httpx.RequestError as e:
                error_msg = f"Content generation request failed: {str(e)}"
                logger.error(error_msg)
                raise Exception(error_msg)
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get task status from mktagent.
        Uses mktagent's existing /tasks/{task_id} endpoint from main.py:591-647
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(
                    f"{self.base_url}/tasks/{task_id}"
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"Retrieved task {task_id} status: {result.get('status')}")
                    return result
                elif response.status_code == 404:
                    raise Exception(f"Task {task_id} not found in mktagent")
                else:
                    error_msg = f"Task status API error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    raise Exception(error_msg)
                    
            except httpx.RequestError as e:
                error_msg = f"Task status request failed: {str(e)}"
                logger.error(error_msg)
                raise Exception(error_msg)
    
    async def submit_research_job(
        self, 
        query: str, 
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Submit research job to mktagent's research API.
        Uses mktagent's existing /api/research/ endpoint
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            research_data = {
                "parameters": {
                    "query": query,
                    "deerflow_config": parameters
                }
            }
            
            try:
                response = await client.post(
                    f"{self.base_url}/api/research/",
                    json=research_data
                )
                
                if response.status_code == 202:  # Research API returns 202
                    result = response.json()
                    logger.info(f"Successfully submitted research job: {result.get('id')}")
                    return result
                else:
                    error_msg = f"Research API error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    raise Exception(error_msg)
                    
            except httpx.RequestError as e:
                error_msg = f"Research submission request failed: {str(e)}"
                logger.error(error_msg)
                raise Exception(error_msg)
    
    async def health_check(self) -> bool:
        """Check if mktagent API is accessible"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/campaigns?limit=1")
                return response.status_code == 200
        except Exception as e:
            logger.warning(f"MktAgent health check failed: {e}")
            return False