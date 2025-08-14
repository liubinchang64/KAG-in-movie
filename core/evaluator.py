import json
import re
from textwrap import dedent
import time

DEFAULT_SCORE_FIELDS = [
    ("relevance", "回答是否切题"),
    ("detail", "是否包含具体电影信息（如名称、评分、风格）"),
    ("truth", "是否内容真实，无编造"),
    ("logic", "是否语言流畅、有条理"),
    ("context_usage", "是否充分利用提供的上下文信息")  # 新增维度
]

DEFAULT_SCORE_THRESHOLD = 20  # 5个维度满分25，20分为及格线


def extract_json_block(s: str) -> str:
    """从字符串中提取JSON代码块"""
    # 尝试匹配标准JSON块
    match = re.search(r"```json(.*?)```", s, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # 尝试匹配没有语言标记的代码块
    match = re.search(r"```(.*?)```", s, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # 尝试匹配可能的JSON结构
    match = re.search(r"\{.*?\}", s, re.DOTALL)
    if match:
        return match.group(0).strip()
    
    return s.strip()


def make_eval_prompt(question, answer, score_fields=None):
    """修复JSON模板尾部逗号问题并添加量化评分标准"""
    score_fields = score_fields or DEFAULT_SCORE_FIELDS
    # 量化评分标准
    score_standards = """
    评分标准（每项1-5分）：
    - 相关性：1分（完全偏离问题），3分（部分相关），5分（完全贴合问题）
    - 细节：1分（无具体信息），3分（含部分信息如名称），5分（含名称、评分、年份等完整信息）
    - 真实性：1分（存在编造内容），3分（部分信息不准确），5分（完全真实无编造）
    - 逻辑：1分（混乱无条理），3分（基本流畅），5分（流畅有条理）
    - 上下文使用：1分（未使用上下文），3分（部分使用），5分（充分利用上下文信息）
    """
    fields_str = "\n".join([
        f"{i + 1}. {k}：{desc}"
        for i, (k, desc) in enumerate(score_fields)
    ])
    # 动态生成字段，无尾部逗号
    fields_json = ",\n  ".join([f"\"{k}\": 5" for k, _ in score_fields])
    eval_prompt = dedent(f"""
    你是一个电影问答质量评估助手。
    {score_standards}
    请从以下维度对回答打分（每项满分5分）：
    {fields_str}
    若得分低于及格线，请提供改进建议和改写版本。
    **严格只输出如下JSON格式，用```json包裹，无多余文字！**
    ```json
    {{
      {fields_json}
      {'' if not fields_json else ','}  # 仅当有字段时才加逗号
      "suggestion": "建议改进点",
      "rewrite": "改写后的答案（可选）"
    }}
    问题：{question}
    回答：{answer}""").strip ()
    return eval_prompt


def evaluate_answer(llm, question: str, answer: str, contexts: list = None,
                    score_fields=None, score_threshold=None) -> dict:
    """评估回答质量（增强版，支持上下文评估）"""
    score_fields = score_fields or DEFAULT_SCORE_FIELDS
    score_threshold = score_threshold or DEFAULT_SCORE_THRESHOLD
    eval_prompt = make_eval_prompt(question, answer, score_fields)
    messages = [
        {"role": "system", "content": "你是评分和反馈助手"},
        {"role": "user", "content": eval_prompt}
    ]

    # 调用LLM进行评估
    try:
        result = llm.call(messages)
        result_content = result["content"] if isinstance(result, dict) else result
        result_json = extract_json_block(result_content)
    except Exception as e:
        print(f"⚠️ LLM评估调用失败：{e}")
        # 构建默认结果
        base = {k: 1 for k, _ in score_fields}
        base.update({
            "suggestion": "LLM评估调用失败",
            "rewrite": ""
        })
        return base

    # 解析评估结果
    try:
        parsed = json.loads(result_json)
        # 补全缺失字段（容错）
        for k, _ in score_fields:
            if k not in parsed:
                parsed[k] = 1  # 缺失字段按最低分处理
        parsed.setdefault("suggestion", "")
        parsed.setdefault("rewrite", "")

        # 额外检查：如果提供了上下文，保持LLM的上下文使用评分
        if contexts and "context_usage" in parsed:
            # 简化处理，保持LLM的原始评分
            pass

        return parsed
    except Exception as e:
        print(f"⚠️ 解析评分JSON失败：{e}")
        print(f"原始输出：{repr(result_content)}")
        # 构建默认结果
        base = {k: 1 for k, _ in score_fields}
        base.update({
            "suggestion": "评分JSON解析失败",
            "rewrite": ""
        })
        return base


def get_checked_answer(
        llm, question: str, messages: list, contexts: list = None,
        score_fields=None, score_threshold=None, show_think=False
) -> str:
    """检查回答质量并返回最优版本（增强版，支持上下文评估）"""
    try:
        if show_think:
            response = llm.call(messages, return_think=True)
            raw_answer = response["content"]
            think = response["think"]
            if think:
                print(f"\n🧠 LLM思考过程：\n{think}\n")
        else:
            response = llm.call(messages)
            raw_answer = response["content"] if isinstance(response, dict) else response

        eval_result = evaluate_answer(
            llm, question, raw_answer, contexts,
            score_fields=score_fields,
            score_threshold=score_threshold
        )

        # 计算总分
        total = sum([
            eval_result.get(k, 0)
            for k, _ in (score_fields or DEFAULT_SCORE_FIELDS)
        ])
        pass_threshold = score_threshold or DEFAULT_SCORE_THRESHOLD
        print(f"🧪 回答质量评分：{total}/{len(score_fields or DEFAULT_SCORE_FIELDS) * 5}")
        print(f"📌 改进建议：{eval_result.get('suggestion', '无')}")

        # 决定返回原始回答还是改写版本
        if total < pass_threshold and eval_result.get("rewrite"):
            print("⚠️ 回答质量较低，使用自动改写版本")
            return eval_result["rewrite"]
        else:
            # 额外优化：如果回答过长，进行精简
            if len(raw_answer) > 1000:
                print("🔄 回答过长，进行精简")
                # 调用LLM进行精简
                shorten_prompt = f"请精简以下回答，保留核心信息，控制在500字以内：\n{raw_answer}"
                shorten_messages = [
                    {"role": "system", "content": "你是文本精简助手"},
                    {"role": "user", "content": shorten_prompt}
                ]
                try:
                    shorten_response = llm.call(shorten_messages)
                    return shorten_response["content"] if isinstance(shorten_response, dict) else shorten_response
                except Exception as e:
                    print(f"⚠️ 精简回答失败：{e}")
            return raw_answer
    except Exception as e:
        print(f"⚠️ 回答检查失败：{e}")
        # 尝试直接返回原始回答，而不是固定错误消息
        try:
            if 'raw_answer' in locals() and raw_answer:
                return raw_answer
        except:
            pass
        return "抱歉，无法生成符合质量的回答"


import logging
import re
from typing import Dict, Any, List, Tuple
from core.llm_service import LLMService
from core.utils import format_time

# 初始化日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("system.log", encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 定义评分标准
SCORING_CRITERIA = [
    {
        "id": "relevance",
        "name": "相关性",
        "description": "回答内容与用户问题的相关程度",
        "weight": 0.3,
        "levels": [
            {"score": 0, "description": "完全不相关，未涉及问题核心"},
            {"score": 1, "description": "部分相关，但未回答主要问题"},
            {"score": 2, "description": "大部分相关，基本回答了主要问题"},
            {"score": 3, "description": "完全相关，精准回答了所有核心问题"}
        ]
    },
    {
        "id": "accuracy",
        "name": "准确性",
        "description": "回答内容的事实准确性",
        "weight": 0.3,
        "levels": [
            {"score": 0, "description": "存在严重事实错误"},
            {"score": 1, "description": "存在部分事实错误"},
            {"score": 2, "description": "基本准确，无重大事实错误"},
            {"score": 3, "description": "完全准确，所有信息都有可靠依据"}
        ]
    },
    {
        "id": "completeness",
        "name": "完整性",
        "description": "回答内容的全面程度",
        "weight": 0.2,
        "levels": [
            {"score": 0, "description": "回答不完整，遗漏关键信息"},
            {"score": 1, "description": "部分完整，遗漏少量关键信息"},
            {"score": 2, "description": "基本完整，包含大部分关键信息"},
            {"score": 3, "description": "完全完整，包含所有相关关键信息"}
        ]
    },
    {
        "id": "clarity",
        "name": "清晰度",
        "description": "回答的组织和表达清晰度",
        "weight": 0.1,
        "levels": [
            {"score": 0, "description": "表达混乱，难以理解"},
            {"score": 1, "description": "部分清晰，存在歧义"},
            {"score": 2, "description": "基本清晰，易于理解"},
            {"score": 3, "description": "非常清晰，结构严谨，表达流畅"}
        ]
    },
    {
        "id": "objectivity",
        "name": "客观性",
        "description": "回答的客观中立程度",
        "weight": 0.1,
        "levels": [
            {"score": 0, "description": "主观臆断，缺乏依据"},
            {"score": 1, "description": "部分主观，依据不足"},
            {"score": 2, "description": "基本客观，有一定依据"},
            {"score": 3, "description": "完全客观，依据充分"}
        ]
    }
]


def generate_evaluation_prompt(question: str, answer: str) -> str:
    """
    生成评估提示
    
    Args:
        question: 用户问题
        answer: 生成的回答
    
    Returns:
        str: 评估提示
    """
    criteria_text = "\n".join([
        f"{i+1}. {criterion['name']} ({criterion['weight']*100}%): {criterion['description']}\n" + \
        "   评分标准:\n" + \
        "   \n".join([f"   - {level['score']}分: {level['description']}" for level in criterion['levels']])
        for i, criterion in enumerate(SCORING_CRITERIA)
    ])

    prompt = f"你是一个专业的问答评估专家。请根据以下评分标准评估回答的质量。\n\n"
    prompt += f"用户问题: {question}\n\n"
    prompt += f"生成的回答: {answer}\n\n"
    prompt += f"评分标准:\n{criteria_text}\n\n"
    prompt += "要求:\n"
    prompt += "1. 严格按照评分标准评估，每个维度给出具体分数和简短理由\n"
    prompt += "2. 计算加权总分（保留两位小数）\n"
    prompt += "3. 最后给出总体评价和改进建议\n"
    prompt += "4. 输出格式必须为JSON，包含以下字段:\n"
    prompt += "   - scores: 各维度评分的字典（键为维度id，值为分数）\n"
    prompt += "   - reasons: 各维度评分理由的字典（键为维度id，值为理由）\n"
    prompt += "   - total_score: 加权总分\n"
    prompt += "   - overall_evaluation: 总体评价\n"
    prompt += "   - improvement_suggestions: 改进建议\n"
    prompt += "5. 确保JSON格式正确，可被Python的json.loads解析\n"

    logger.info("评估提示生成完成")
    return prompt


def parse_evaluation_result(evaluation_text: str) -> Dict[str, Any]:
    """
    解析评估结果
    
    Args:
        evaluation_text: 评估文本
    
    Returns:
        Dict[str, Any]: 解析后的评估结果
    """
    try:
        # 提取JSON部分
        json_match = re.search(r'\{.*\}', evaluation_text, re.DOTALL)
        if not json_match:
            logger.error("评估结果中未找到JSON格式内容")
            return {
                "error": "评估结果格式错误，未找到JSON内容",
                "raw_text": evaluation_text
            }

        json_str = json_match.group()
        # 替换可能的转义问题
        json_str = json_str.replace('\n', '').replace('\r', '')
        # 修复可能的JSON格式问题
        json_str = re.sub(r'([{,])\s*([a-zA-Z0-9_]+)\s*:', r'\1"\2":', json_str)
        json_str = re.sub(r':\s*([a-zA-Z0-9_]+)\s*([},])', r':"\1"\2', json_str)
        json_str = re.sub(r':\s*([a-zA-Z0-9_]+)\s*$', r':"\1"', json_str)

        # 解析JSON
        evaluation = json.loads(json_str)

        # 验证必要字段
        required_fields = ["scores", "reasons", "total_score", "overall_evaluation", "improvement_suggestions"]
        for field in required_fields:
            if field not in evaluation:
                logger.warning(f"评估结果缺少必要字段: {field}")
                evaluation[field] = "" if field != "total_score" else 0

        logger.info(f"评估结果解析成功，总分: {evaluation.get('total_score', 0)}")
        return evaluation
    except json.JSONDecodeError as e:
        logger.error(f"评估结果JSON解析失败: {e}")
        return {
            "error": f"JSON解析失败: {str(e)}",
            "raw_text": evaluation_text
        }
    except Exception as e:
        logger.error(f"评估结果解析异常: {e}")
        return {
            "error": f"解析异常: {str(e)}",
            "raw_text": evaluation_text
        }


def calculate_weighted_score(scores: Dict[str, int]) -> float:
    """
    计算加权总分
    
    Args:
        scores: 各维度评分
    
    Returns:
        float: 加权总分
    """
    total = 0.0
    weight_sum = 0.0
    for criterion in SCORING_CRITERIA:
        criterion_id = criterion["id"]
        weight = criterion["weight"]
        if criterion_id in scores:
            total += scores[criterion_id] * weight
            weight_sum += weight
        else:
            logger.warning(f"缺少维度评分: {criterion_id}")

    if weight_sum == 0:
        logger.error("所有维度权重和为0，无法计算总分")
        return 0.0

    # 归一化到0-10分
    normalized_score = (total / weight_sum) * 10 / 3
    logger.info(f"加权总分计算完成: {normalized_score:.2f}")
    return round(normalized_score, 2)


def get_checked_answer(question: str, answer: str) -> Dict[str, Any]:
    """
    获取评估后的回答
    
    Args:
        question: 用户问题
        answer: 生成的回答
    
    Returns:
        Dict[str, Any]: 评估结果
    """
    try:
        # 延迟导入避免循环依赖
        from core.llm_service import LLMService
        from core.utils import load_config

        # 加载配置
        config = load_config()
        if not config:
            logger.error("加载配置失败，使用默认配置")
            config = {}

        # 初始化LLM服务
        llm = LLMService(config)

        # 生成评估提示
        prompt = generate_evaluation_prompt(question, answer)

        # 调用LLM进行评估
        logger.info("开始调用LLM进行评估")
        t_start = time.time()
        messages = [
            {"role": "system", "content": "你是一个专业的问答评估专家。"},
            {"role": "user", "content": prompt}
        ]
        response = llm.call(messages)
        t_end = time.time()
        eval_time = format_time(t_end - t_start)
        logger.info(f"LLM评估完成，耗时{eval_time}")

        # 解析评估结果
        evaluation_text = response["content"].strip()
        evaluation = parse_evaluation_result(evaluation_text)

        # 如果解析失败，使用备用方法计算分数
        if "error" in evaluation:
            logger.warning("评估结果解析失败，使用备用方法计算分数")
            # 假设一个默认评分
            default_scores = {criterion["id"]: 2 for criterion in SCORING_CRITERIA}
            evaluation = {
                "scores": default_scores,
                "reasons": {criterion["id"]: "评估解析失败，使用默认评分" for criterion in SCORING_CRITERIA},
                "total_score": calculate_weighted_score(default_scores),
                "overall_evaluation": "评估解析失败，使用默认评分",
                "improvement_suggestions": "无法提供具体建议，因为评估解析失败"
            }

        # 添加评估耗时
        evaluation["eval_time"] = eval_time

        return evaluation
    except Exception as e:
        logger.error(f"评估过程发生错误: {e}")
        # 返回错误信息
        return {
            "error": str(e),
            "scores": {},
            "reasons": {},
            "total_score": 0.0,
            "overall_evaluation": "评估失败",
            "improvement_suggestions": "无法提供建议，因为评估失败"
        }
