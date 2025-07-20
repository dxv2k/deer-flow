"""
Research Report Summarization Module
Integrates summarization capabilities into the research workflow
"""

import logging
import traceback
import json
from typing import Dict, Any, List
from pathlib import Path
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langfuse import observe
from langfuse.langchain import CallbackHandler
import re

logger = logging.getLogger(__name__)


class ReportInsight(BaseModel):
    """
    Research report insight schema matching summarize_report.py
    """

    summary: str
    key_points: List[str]


def extract_json_from_markdown(text: str) -> str:
    """
    Extract JSON string from a Markdown code block, if present.
    """
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"```\s*$", "", cleaned)
    return cleaned.strip()


class ResearchSummarizer:
    """
    Summarizes research reports using OpenAI via LangChain
    Integrated version of summarize_report.py for workflow usage
    """

    def __init__(self, model: str = "gpt-4-1106-preview", temperature: float = 0.2):
        """
        Initialize the summarizer with LLM configuration

        Args:
            model: OpenAI model to use (default: gpt-4-1106-preview for gpt-4.1-mini)
            temperature: LLM temperature for consistency
        """
        self.model = model
        self.temperature = temperature

        # Initialize the LLM
        self.llm = ChatOpenAI(model=self.model, temperature=self.temperature)

        # Prepare the prompt template
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert market analyst. Given a research report, extract a very short summary (max 2 sentences) and 3-5 key points. Output only JSON in the following format: {{summary: str, key_points: list[str]}}.",
                ),
                ("human", "{report}"),
            ]
        )
        self.langfuse_handler = CallbackHandler()

        # Create the chain
        self.chain = self.prompt | self.llm

        logger.info(f"ResearchSummarizer initialized with model: {self.model}")

    def summarize_report_text(self, report_text: str) -> ReportInsight:
        """
        Summarize research report text directly

        Args:
            report_text: The full research report text

        Returns:
            ReportInsight: Structured summary and key points

        Raises:
            Exception: If summarization fails
        """
        try:
            logger.info(f"Summarizing report (length: {len(report_text)} chars)")

            # Run the LLM chain
            response = self.chain.invoke(
                {"report": report_text}, config={"callbacks": [self.langfuse_handler]}
            )
            logger.info(f"Raw LLM response: {response.content}")

            # Parse the JSON output
            json_str = extract_json_from_markdown(response.content)
            data = json.loads(json_str)
            insight = ReportInsight(**data)

            logger.info(
                f"Successfully generated summary: {len(insight.summary)} chars, {len(insight.key_points)} key points"
            )
            return insight

        except Exception as e:
            logger.error(f"Error summarizing report: {e}\n{traceback.format_exc()}")
            raise

    def summarize_report_file(self, report_path: str) -> ReportInsight:
        """
        Summarize research report from file path

        Args:
            report_path: Path to the report file

        Returns:
            ReportInsight: Structured summary and key points
        """
        try:
            # Read the report
            report_text = Path(report_path).read_text(encoding="utf-8")
            logger.info(
                f"Loaded report from {report_path} (length: {len(report_text)} chars)"
            )

            return self.summarize_report_text(report_text)

        except Exception as e:
            logger.error(
                f"Error loading and summarizing report file {report_path}: {e}\n{traceback.format_exc()}"
            )
            raise


@observe(name="summarize-research-result", capture_input=True, capture_output=True)
def summarize_research_result(
    final_report: str, model: str = "gpt-4.1-mini", temperature: float = 0.2
) -> Dict[str, Any]:
    """
    Convenience function to summarize research results
    Used by background processor to generate insights

    Args:
        final_report: The completed research report text
        model: OpenAI model to use
        temperature: LLM temperature

    Returns:
        Dict containing summary and insights data

    Example return format:
        {
            "summary": "Short summary text...",
            "insights": ["Key point 1", "Key point 2", ...],
            "success": True,
            "error": None
        }
    """
    try:
        if not final_report or not final_report.strip():
            logger.warning("Empty or missing final_report provided for summarization")
            return {
                "summary": None,
                "insights": [],
                "success": False,
                "error": "Empty report provided",
            }

        # Initialize summarizer
        summarizer = ResearchSummarizer(model=model, temperature=temperature)

        # Generate insights
        insight = summarizer.summarize_report_text(final_report)

        logger.info(
            f"Research summarization successful: {len(insight.summary)} char summary, {len(insight.key_points)} insights"
        )

        return {
            "summary": insight.summary,
            "insights": insight.key_points,
            "success": True,
            "error": None,
        }

    except Exception as e:
        error_msg = f"Research summarization failed: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")

        return {"summary": None, "insights": [], "success": False, "error": error_msg}


# For backward compatibility and testing
def main():
    """
    Test function for command-line usage
    Compatible with original summarize_report.py interface
    """
    import sys

    report_path = sys.argv[1] if len(sys.argv) > 1 else "report.md"

    try:
        summarizer = ResearchSummarizer()
        insight = summarizer.summarize_report_file(report_path)
        print(insight.model_dump_json(indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Failed to summarize report: {e}")


if __name__ == "__main__":
    main()
