"""
RAG Evaluator — evaluates retrieval+generation quality using RAGAS.

Metrics:
- Faithfulness: how factually consistent the answer is with the context
- Answer Relevancy: how relevant the answer is to the question
- Context Precision: how precise the retrieved context is
- Context Recall: how much of the relevant context was retrieved
"""
import json
import os

from datasets import Dataset


class RAGEvaluator:
    """
    Evaluates RAG pipeline quality using the RAGAS framework.

    Test set format (JSON):
    [
        {
            "question": "AI算法工程师的薪资范围是多少？",
            "ground_truth": "2026年AI算法工程师在北京的薪资大致在25-40K...",
            "contexts": ["context chunk 1", "context chunk 2"],
            "answer": "根据数据分析，AI算法工程师在北上深的薪资..."
        },
        ...
    ]

    Recommended test set size: 30+ questions.
    """

    def __init__(self, llm, embeddings):
        self.llm = llm
        self.embeddings = embeddings

    def evaluate(self, test_set_path: str) -> dict:
        """
        Run RAGAS evaluation on a test set.

        Returns dict with metric scores.
        """
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )

        if not os.path.exists(test_set_path):
            raise FileNotFoundError(f"Test set not found: {test_set_path}")

        with open(test_set_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        dataset = Dataset.from_list(test_data)
        results = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm=self.llm,
            embeddings=self.embeddings,
        )
        return results

    def generate_test_report(self, results: dict) -> str:
        """
        Generate a Markdown evaluation report.

        Includes:
        - Overall metrics table
        - Comparison table (vector only, +BM25, +reranker, full pipeline)
        """
        report = f"""# JobSense RAG 评估报告

## 综合指标

| 指标 | 得分 | 说明 |
|------|------|------|
| Faithfulness | {results.get('faithfulness', 'N/A'):.3f} | 答案与上下文的忠实度 |
| Answer Relevancy | {results.get('answer_relevancy', 'N/A'):.3f} | 答案与问题的相关度 |
| Context Precision | {results.get('context_precision', 'N/A'):.3f} | 检索上下文的精确度 |
| Context Recall | {results.get('context_recall', 'N/A'):.3f} | 检索上下文的召回度 |

## 策略对比

| 检索策略 | Faithfulness | Answer Relevancy | Context Precision |
|------|-------------|-----------------|------------------|
| 纯向量检索 | - | - | - |
| (+BM25) 混合检索 | - | - | - |
| + RRF 融合 | - | - | - |
| + 重排序 (完整管线) | - | - | - |

---
*注: 完整的策略对比需运行 ablations 脚本获取。*
"""
        return report
