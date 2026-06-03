"""
Evaluation Route — trigger RAGAS evaluation runs.

Runs the evaluation pipeline and returns a Markdown report.
"""
import logging
from fastapi import APIRouter, HTTPException

from api.schemas import EvaluationRequest, EvaluationResponse
from config import TEST_SET_PATH, LLM_MODEL

logger = logging.getLogger(__name__)

router = APIRouter()

# Global references — set by main.py
_evaluator = None


def init_evaluation_route(evaluator):
    """Initialize evaluation route with the RAG evaluator."""
    global _evaluator
    _evaluator = evaluator


@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_system(request: EvaluationRequest):
    """
    Run a RAGAS evaluation on the test set.

    Returns a Markdown report with faithfulness, relevancy,
    precision, and recall metrics.
    """
    if _evaluator is None:
        raise HTTPException(status_code=503, detail="Evaluator not initialized")

    try:
        test_set_path = request.test_set_path or str(TEST_SET_PATH)
        results = _evaluator.evaluate(test_set_path)
        report = _evaluator.generate_test_report(results)

        return EvaluationResponse(
            status="success",
            report=report,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Evaluation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
