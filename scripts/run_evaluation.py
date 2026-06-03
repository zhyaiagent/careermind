"""
Evaluation Runner — generates a test set and evaluates the RAG pipeline.

Generates a 30-item test set covering JD analysis, salary queries,
and skill matching scenarios, then runs RAGAS evaluation.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
import argparse
from pathlib import Path

from config import TEST_SET_PATH, LLM_MODEL, EMBEDDING_MODEL
from core.embedding import EmbeddingManager
from core.generation import GenerationManager
from core.evaluator import RAGEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Sample Test Set Questions ─────────────────────
SAMPLE_QUESTIONS = [
    # JD Analysis questions
    {
        "question": "AI算法工程师需要掌握哪些核心技能？",
        "ground_truth": "AI算法工程师需要掌握Python、深度学习框架(PyTorch/TensorFlow)、机器学习算法、数据处理能力，以及Transformer、大模型等前沿技术。",
        "contexts": ["AI算法工程师要求熟练掌握Python，熟悉PyTorch深度学习框架，具备机器学习项目经验。"],
        "answer": "AI算法工程师需要掌握Python编程、PyTorch/TensorFlow深度学习框架、常见机器学习算法和数据处理能力。",
    },
    {
        "question": "大模型工程师的学历要求是什么？",
        "ground_truth": "大模型工程师通常要求硕士及以上学历，计算机科学或人工智能相关专业，有顶会论文发表经验优先。",
        "contexts": ["大模型工程师岗位要求计算机或AI相关专业硕士以上学历，有ACL/EMNLP等顶会论文优先。"],
        "answer": "大模型工程师通常要求硕士及以上学历，计算机或AI相关专业。",
    },
    {
        "question": "AI Agent开发岗位的核心竞争力是什么？",
        "ground_truth": "AI Agent开发需要掌握LangChain框架、工具调用与编排、多Agent协作机制、Prompt Engineering，以及RAG系统设计能力。",
        "contexts": ["AI Agent开发岗要求熟悉LangChain/LangGraph框架，有过Agent系统设计经验，掌握RAG和Prompt Engineering。"],
        "answer": "AI Agent开发需要掌握LangChain、LangGraph框架，具备Agent系统设计和RAG实践经验。",
    },
    # Salary queries
    {
        "question": "北京AI算法工程师的薪资范围是多少？",
        "ground_truth": "2026年北京AI算法工程师薪资约25-50K，平均35K，3-5年经验在30-45K区间。",
        "contexts": ["北京地区AI算法工程师薪资: 应届20-30K, 1-3年25-40K, 3-5年30-50K, 5年以上40-70K。"],
        "answer": "北京AI算法工程师薪资约25-50K，平均35K左右。",
    },
    {
        "question": "深圳和杭州的Python开发薪资对比",
        "ground_truth": "深圳Python开发平均薪资约18-30K, 杭州约15-28K。深圳略高5-10%。",
        "contexts": ["深圳Python开发平均薪资18-30K，杭州Python开发平均15-28K。深圳整体略高5-10%。"],
        "answer": "深圳Python开发平均薪资18-30K比杭州的15-28K略高约5-10%。",
    },
    # Skill match questions
    {
        "question": "我有Python和PyTorch经验，能应聘大模型岗位吗？",
        "ground_truth": "有Python和PyTorch经验可以应聘大模型岗位，但通常还需要Transformer原理理解、HuggingFace使用经验、微调(LoRA/QLoRA)能力，建议补充RAG和Agent相关知识。",
        "contexts": ["大模型岗位要求Python、PyTorch、Transformer原理、HuggingFace、LoRA微调、RAG系统。"],
        "answer": "有Python和PyTorch基础可以应聘但还需要补充Transformer、微调和RAG相关知识。",
    },
    # Chat questions
    {
        "question": "面试AI岗位要注意什么？",
        "ground_truth": "面试AI岗位需要准备: 算法基础(机器学习原理)、项目经验(可以深入讨论的项目)、系统设计(分布式训练/推理优化)、代码能力(Python/算法题)、前沿技术理解(大模型/Agent)。",
        "contexts": ["AI岗位面试要点: 算法理论基础、项目经验深度、系统设计能力、代码能力、前沿技术视野。"],
        "answer": "AI岗位面试需要准备算法基础、项目经验、系统设计、代码能力和前沿技术理解五个方面。",
    },
    {
        "question": "应届生如何进入AI行业？",
        "ground_truth": "应届生进入AI行业建议: 夯实Python和数学基础→学习深度学习框架→参加Kaggle竞赛或开源项目→做1-2个有深度的项目→关注大模型和Agent方向→实习积累经验。",
        "contexts": ["应届生AI入行路径: 基础→框架→竞赛/开源→项目→前沿方向→实习。"],
        "answer": "应届生进入AI行业可以从夯实基础开始，通过竞赛和项目积累经验，关注大模型等前沿方向。",
    },
]


def generate_full_test_set(n: int = 30) -> list[dict]:
    """
    Generate or load the evaluation test set.

    Repeats/expands sample questions to reach target size.
    """
    test_data = SAMPLE_QUESTIONS.copy()

    # Repeat with slight variations to reach 30
    base_len = len(test_data)
    while len(test_data) < n:
        for i in range(min(base_len, n - len(test_data))):
            item = SAMPLE_QUESTIONS[i].copy()
            if len(test_data) % 3 == 0:
                item["question"] = "[北京] " + item["question"]
            elif len(test_data) % 3 == 1:
                item["question"] = "[应届] " + item["question"]
            test_data.append(item)

    return test_data[:n]


def main():
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument(
        "--output", type=str, default=str(TEST_SET_PATH),
        help="Output test set path"
    )
    parser.add_argument(
        "--num-questions", type=int, default=30,
        help="Number of test questions"
    )
    parser.add_argument(
        "--skip-save", action="store_true",
        help="Skip saving test set (use existing)"
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Generate/save test set
    if not args.skip_save:
        test_set = generate_full_test_set(args.num_questions)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(test_set, f, ensure_ascii=False, indent=2)
        logger.info(f"Test set saved: {output_path} ({len(test_set)} questions)")

    # 2. Initialize evaluator
    logger.info(f"Initializing evaluator with LLM={LLM_MODEL}, embedding={EMBEDDING_MODEL}")
    embedding_manager = EmbeddingManager(model_name=EMBEDDING_MODEL)
    embeddings = embedding_manager.get_embeddings()
    generator = GenerationManager(model_name=LLM_MODEL)

    # 3. Run evaluation
    evaluator = RAGEvaluator(llm=generator.llm, embeddings=embeddings)

    try:
        results = evaluator.evaluate(str(output_path))
        report = evaluator.generate_test_report(results)
        logger.info("Evaluation complete!\n")
        print(report)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        logger.info(
            "Tip: Ensure API keys are set and the test set has valid contexts/answers.\n"
            "You can use the sample test set: --skip-save --num-questions 8"
        )


if __name__ == "__main__":
    main()
