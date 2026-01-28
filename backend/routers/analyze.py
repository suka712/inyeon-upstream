from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import ValidationError

from backend.core.dependencies import get_ollama_client
from backend.models.schemas import AnalyzeRequest, AnalyzeResponse
from backend.prompts.analyze_prompt import build_analyze_prompt
from backend.services.ollama_client import OllamaClient, OllamaError


router = APIRouter()


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Analyze git diff",
    description="Analyze a git diff and return structured insights about the changes.",
)
async def analyze_diff(
    request: AnalyzeRequest,
    ollama: OllamaClient = Depends(get_ollama_client),
) -> AnalyzeResponse:
    """
    Analyze a git diff and return:
    - Summary of changes
    - Impact assessment (low/medium/high)
    - Categories (feat, fix, refactor, etc.)
    - Breaking changes
    - Security concerns
    - Per-file change details
    """
    prompt = build_analyze_prompt(request.diff, request.context)

    try:
        result = await ollama.generate(prompt, json_mode=True)
    except OllamaError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM service error: {e}",
        )

    try:
        return AnalyzeResponse(**result)
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM returned invalid response: {e.error_count()} validation errors",
        )
