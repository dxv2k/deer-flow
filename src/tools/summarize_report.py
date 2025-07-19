import logging
import traceback
import json
from pathlib import Path
from typing import Any, Dict
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ReportInsight(BaseModel):
    summary: str
    key_points: list[str]


def extract_json_from_markdown(text: str) -> str:
    """
    Extract JSON string from a Markdown code block, if present.
    """
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"```\s*$", "", cleaned)
    return cleaned.strip()


def summarize_report_md(report_path: str) -> ReportInsight:
    """
    Summarize the report.md file using OpenAI (gpt-4.1-mini) via LangChain and return a structured JSON.
    """
    try:
        # Read the report
        report_text = Path(report_path).read_text(encoding="utf-8")
        logging.info(f"Loaded report from {report_path} (length: {len(report_text)} chars)")

        # Prepare the prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert market analyst. Given a research report, extract a very short summary (max 2 sentences) and 3-5 key points. Output only JSON in the following format: {{summary: str, key_points: list[str]}}."),
            ("human", "{report}")
        ])

        # Initialize the LLM
        llm = ChatOpenAI(model="gpt-4-1106-preview", temperature=0.2)
        chain = prompt | llm

        # Run the chain
        response = chain.invoke({"report": report_text})
        logging.info(f"Raw LLM response: {response.content}")

        # Parse the JSON output
        json_str = extract_json_from_markdown(response.content)
        data = json.loads(json_str)
        insight = ReportInsight(**data)
        return insight
    except Exception as e:
        logging.error(f"Error summarizing report: {e}\n{traceback.format_exc()}")
        raise


def main():
    import sys
    report_path = sys.argv[1] if len(sys.argv) > 1 else "report.md"
    try:
        insight = summarize_report_md(report_path)
        print(insight.model_dump_json(indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Failed to summarize report: {e}")


if __name__ == "__main__":
    main() 