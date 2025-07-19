#!/usr/bin/env python3
"""
DeerFlow Research API Client Example

This example demonstrates how to use the DeerFlow Deep Research API
to submit research queries, poll for results, and retrieve data.

Usage:
    python examples/research_client.py
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any
import httpx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DeerFlowResearchClient:
    """
    Python client for DeerFlow Research API
    """
    
    # def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
    def __init__(self, base_url: str = "http://100.124.29.25:8009", timeout: int = 30):
        """
        Initialize the research client
        
        Args:
            base_url: Base URL of the DeerFlow API server
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.api_base = f"{self.base_url}/api/research"
        self.timeout = timeout
        
        logger.info(f"Initialized DeerFlow Research Client: {self.api_base}")
    
    async def create(
        self,
        query: str,
        max_plan_iterations: int = 1,
        max_step_num: int = 3,
        enable_background_investigation: bool = True,
        mktagent_campaign_id: Optional[int] = None,
        submit_to_mktagent: bool = False
    ) -> Dict[str, Any]:
        """
        Submit a research query for processing
        
        Args:
            query: Research question or topic to investigate
            max_plan_iterations: Maximum planning iterations (1-5)
            max_step_num: Maximum research steps (1-10)
            enable_background_investigation: Enable deep research
            mktagent_campaign_id: Optional MktAgent campaign ID
            submit_to_mktagent: Auto-submit to MktAgent
            
        Returns:
            Dict containing research_id, status, and message
        """
        payload = {
            "query": query,
            "max_plan_iterations": max_plan_iterations,
            "max_step_num": max_step_num,
            "enable_background_investigation": enable_background_investigation,
            "submit_to_mktagent": submit_to_mktagent
        }
        
        if mktagent_campaign_id:
            payload["mktagent_campaign_id"] = mktagent_campaign_id
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.api_base}/submit",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                result = response.json()
                
                logger.info(f"Research submitted successfully: {result['research_id']}")
                return result
                
        except httpx.HTTPError as e:
            logger.error(f"HTTP error submitting research: {e}")
            raise
        except Exception as e:
            logger.error(f"Error submitting research: {e}")
            raise
    
    async def get(self, research_id: str) -> Dict[str, Any]:
        """
        Get the status and results of a research task
        
        Args:
            research_id: Unique research task identifier
            
        Returns:
            Dict containing complete research results and metadata
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.api_base}/status/{research_id}")
                response.raise_for_status()
                result = response.json()
                
                logger.info(f"Retrieved status for {research_id}: {result['status']}")
                return result
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.error(f"Research task {research_id} not found")
            else:
                logger.error(f"HTTP {e.response.status_code} error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error getting research status: {e}")
            raise
    
    async def get_results(self, research_id: str) -> Dict[str, Any]:
        """
        Get full research results for a completed task
        
        Args:
            research_id: Unique research task identifier
            
        Returns:
            Dict containing full report, summary, insights, and metadata
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.api_base}/results/{research_id}")
                response.raise_for_status()
                result = response.json()
                
                logger.info(f"Retrieved full results for {research_id}")
                return result
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.error(f"Research task {research_id} not found")
            elif e.response.status_code == 400:
                logger.error(f"Research task {research_id} not completed yet")
            else:
                logger.error(f"HTTP {e.response.status_code} error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error getting research results: {e}")
            raise
    
    async def list(
        self,
        page: int = 1,
        limit: int = 10,
        status: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        List research tasks with pagination
        
        Args:
            page: Page number (1-based)
            limit: Items per page (1-100)
            status: Filter by status (pending, running, completed, failed)
            
        Returns:
            Dict containing data array, total count, page, and limit
        """
        params = {"page": page, "limit": limit}
        if status:
            params["status"] = status
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.api_base}/list", params=params)
                response.raise_for_status()
                result = response.json()
                
                logger.info(f"Retrieved {len(result['data'])} tasks (page {page}, total {result['total']})")
                return result
                
        except httpx.HTTPError as e:
            logger.error(f"HTTP error listing research tasks: {e}")
            raise
        except Exception as e:
            logger.error(f"Error listing research tasks: {e}")
            raise
    
    async def poll(
        self,
        research_id: str,
        poll_interval: int = 5,
        max_wait_time: int = 900
    ) -> Dict[str, Any]:
        """
        Poll a research task until completion or timeout
        
        Args:
            research_id: Unique research task identifier
            poll_interval: Seconds between status checks
            max_wait_time: Maximum time to wait in seconds
            
        Returns:
            Dict containing final research results
        """
        start_time = time.time()
        
        logger.info(f"Starting to poll {research_id} (max wait: {max_wait_time}s)")
        
        while time.time() - start_time < max_wait_time:
            try:
                status_data = await self.get(research_id)
                current_status = status_data['status']
                
                logger.info(f"Task {research_id} status: {current_status}")
                
                if current_status == 'completed':
                    logger.info(f"Task {research_id} completed successfully!")
                    return status_data
                elif current_status == 'failed':
                    error_msg = status_data.get('error', 'Unknown error')
                    logger.error(f"Task {research_id} failed: {error_msg}")
                    raise Exception(f"Research task failed: {error_msg}")
                elif current_status in ['pending', 'running']:
                    logger.info(f"Task {research_id} still {current_status}, waiting {poll_interval}s...")
                    await asyncio.sleep(poll_interval)
                else:
                    logger.warning(f"Unknown status '{current_status}' for task {research_id}")
                    await asyncio.sleep(poll_interval)
                    
            except httpx.HTTPError as e:
                logger.error(f"HTTP error during polling: {e}")
                await asyncio.sleep(poll_interval)
            except Exception as e:
                logger.error(f"Error during polling: {e}")
                raise
        
        # Timeout reached
        elapsed = time.time() - start_time
        logger.error(f"Polling timeout after {elapsed:.1f}s for task {research_id}")
        raise TimeoutError(f"Research task did not complete within {max_wait_time} seconds")
    
    async def delete(self, research_id: str) -> Dict[str, Any]:
        """
        Delete a research task
        
        Args:
            research_id: Unique research task identifier
            
        Returns:
            Dict containing success message
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.delete(f"{self.api_base}/tasks/{research_id}")
                response.raise_for_status()
                result = response.json()
                
                logger.info(f"Deleted research task {research_id}")
                return result
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.error(f"Research task {research_id} not found")
            else:
                logger.error(f"HTTP {e.response.status_code} error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error deleting research task: {e}")
            raise


async def example_usage():
    """
    Example usage of the DeerFlow Research Client
    """
    # Initialize client (adjust URL for your server)
    client = DeerFlowResearchClient()
    
    try:
        # Example 1: Create a research query
        logger.info("=== Example 1: Create Research Task ===")
        research_query = "Analyze the future trends in artificial intelligence and machine learning for 2025-2030"
        
        create_result = await client.create(
            query=research_query,
            max_plan_iterations=2,
            max_step_num=4,
            enable_background_investigation=True
        )
        
        research_id = create_result['research_id']
        print(f"✅ Created research: {research_id}")
        print(f"   Status: {create_result['status']}")
        print(f"   Message: {create_result['message']}")
        
        # Example 2: Poll until completion
        logger.info("=== Example 2: Poll for Completion ===")
        completed_result = await client.poll(
            research_id=research_id,
            poll_interval=3,
            max_wait_time=180  # 3 minutes
        )
        
        print(f"✅ Research completed!")
        print(f"   Processing time: {completed_result.get('processing_time', 'N/A')} seconds")
        print(f"   Sources analyzed: {completed_result.get('sources_analyzed', 'N/A')}")
        
        # Example 3: Get full results
        logger.info("=== Example 3: Get Full Results ===")
        full_results = await client.get_results(research_id)
        
        print(f"✅ Full results retrieved:")
        print(f"   Summary: {full_results.get('summary', 'N/A')[:100]}...")
        print(f"   Insights count: {len(full_results.get('insights', []))}")
        print(f"   Report length: {len(full_results.get('final_report', ''))} characters")
        
        # Example 4: List all research tasks
        logger.info("=== Example 4: List Research Tasks ===")
        task_list = await client.list(page=1, limit=5)
        
        print(f"✅ Found {task_list['total']} total tasks:")
        for i, task in enumerate(task_list['data'][:3], 1):
            print(f"   {i}. {task['research_id']}: {task['status']} - {task['query'][:50]}...")
            if task.get('summary'):
                print(f"      Summary: {task['summary'][:80]}...")
            print(f"      Created: {task['created_at']}")
        
        # Example 5: Clean up (optional)
        logger.info("=== Example 5: Cleanup (Optional) ===")
        # Uncomment to delete the test task
        # delete_result = await client.delete(research_id)
        # print(f"✅ Deleted task: {delete_result['message']}")
        
        print("\n🎉 All examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    # Run the example
    asyncio.run(example_usage()) 
