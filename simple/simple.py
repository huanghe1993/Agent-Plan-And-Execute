from openai import OpenAI
import json
import os
import re

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ======================
# 1. LLM 调用（默认 DeepSeek，可选 Qwen）
# ======================
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def _get_llm_client():
    """默认 DeepSeek；设置 LLM_PROVIDER=qwen 时使用通义千问。"""
    provider = (os.environ.get("LLM_PROVIDER") or "deepseek").strip().lower()
    if provider == "qwen":
        key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
        base = os.environ.get("DASHSCOPE_BASE_URL") or DASHSCOPE_BASE_URL
        model = os.environ.get("LLM_MODEL") or "qwen-turbo"
    else:
        key = os.environ.get("DEEPSEEK_API_KEY")
        base = os.environ.get("DEEPSEEK_BASE_URL") or DEEPSEEK_BASE_URL
        model = os.environ.get("LLM_MODEL") or "deepseek-chat"
    return OpenAI(api_key=key, base_url=base), model


_client, _model = _get_llm_client()


def llm(prompt):
    response = _client.chat.completions.create(
        model=_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()

# ======================
# 2. 工具库（真实可调用）
# ======================
def calculator(expression):
    """计算器：支持加减乘除"""
    try:
        return f"计算结果：{eval(expression)}"
    except:
        return "计算错误"

def search(query):
    """模拟搜索（你可以换成真实搜索API）"""
    return f"搜索结果：关于 {query} 的相关信息"

def get_weather(city):
    """模拟天气"""
    return f"{city} 天气：晴，25℃"

# 工具映射
TOOL_MAP = {
    "calculator": calculator,
    "search": search,
    "get_weather": get_weather,
}

# ======================
# 3. 混合 Agent：Plan + ReAct + Tool + Replan
# ======================
class PlanReactToolAgent:
    def __init__(self):
        self.plan = []
        self.current_step = 0
        self.fail_count = 0
        self.max_retry = 2
        self.max_fail_before_replan = 3

    # --------------------
    # 1. 制定计划（只输出“待执行步骤”，不包含答案或结果）
    # --------------------
    def _normalize_plan_step(self, line):
        """去掉行首编号，只保留步骤描述；若有冒号后的结果描述则截断。"""
        line = line.strip()
        # 去掉 "1. " / "1、" / "第一步 " 等
        line = re.sub(r"^\d+[\.\)、]\s*", "", line)
        line = re.sub(r"^第[一二三四五六七八九十\d]+步\s*", "", line)
        # 只保留“步骤描述”，不要“：结果”（计划阶段不应包含执行结果）
        if "：" in line:
            line = line.split("：", 1)[0].strip()
        if ":" in line and not line.startswith("http"):
            part = line.split(":", 1)[0].strip()
            if len(part) < len(line) and len(part) <= 30:
                line = part
        return line.strip()

    def make_plan(self, task):
        prompt = f"""任务：{task}

        请只输出「待执行的步骤」列表，每行一步。要求：
        - 每行仅写这一步要做什么（例如：查询北京天气、计算 123+456、总结），不要写执行结果、不要写答案或结论。
        - 用数字编号，如：1. 步骤一
        - 不要包含冒号后面的结果描述，不要提前写出天气、数值等结果。
        """
        plan_text = llm(prompt)
        raw_lines = [s.strip() for s in plan_text.split("\n") if s.strip()]
        self.plan = [self._normalize_plan_step(line) for line in raw_lines if self._normalize_plan_step(line)]
        self.current_step = 0
        self.fail_count = 0
        print("\n📋 计划：")
        for i, step in enumerate(self.plan):
            print(f"{i+1}. {step}")
        print("-"*50)

    # --------------------
    # 2. ReAct + 工具调用
    # --------------------
    def react_and_act(self, task, step, observation):
        prompt = f"""
                    任务：{task}
                    当前步骤：{step}
                    已有信息：{observation}

                    你可以使用工具：
                    - search(query)
                    - calculator(expression)
                    - get_weather(city)

                    请严格输出 JSON 格式：
                    {{
                        "think": "思考",
                        "tool": "工具名",
                        "args": "参数"
                    }}
                    只输出 JSON，不要其他文字。
                    """
        try:
            res = llm(prompt)
            data = json.loads(res)
            tool = data.get("tool")
            args = data.get("args")
            think = data.get("think")

            print(f"🤔 思考：{think}")
            print(f"🔧 调用工具：{tool}({args})")

            # 执行工具
            if tool in TOOL_MAP:
                tool_result = TOOL_MAP[tool](args)
            else:
                tool_result = "工具不存在"

            return tool_result

        except Exception as e:
            return f"执行失败：{str(e)}"

    # --------------------
    # 3. 是否需要 replan
    # --------------------
    def need_replan(self, step, result):
        if self.fail_count >= self.max_fail_before_replan:
            return True

        if "错误" in result or "失败" in result or "不存在" in result:
            return True

        prompt = f"""
                判断结果是否对步骤有帮助：
                步骤：{step}
                结果：{result}
                只回答 是/否
                """
        answer = llm(prompt).strip()
        helpful = "有帮助" if answer == "是" else "无帮助"
        need_replan = False
        if answer == "否":
            need_replan = True
            print(f"重规划结果是否对步骤有帮助：{helpful}")
        return need_replan

    # --------------------
    # 4. 单步执行（带重试）
    # --------------------
    def run_step(self, task, step):
        obs = "开始执行"
        for _ in range(self.max_retry + 1):
            result = self.react_and_act(task, step, obs)
            print(f"✅ 结果：{result}\n")

            if "错误" not in result and "失败" not in result:
                self.fail_count = 0
                return result

            self.fail_count += 1
            print(f"❌ 失败，累计失败：{self.fail_count}")

        return result

    # --------------------
    # 5. 主运行流程
    # --------------------
    def run(self, task):
        print(f"🎯 任务：{task}\n")
        self.make_plan(task)

        while self.current_step < len(self.plan):
            step = self.plan[self.current_step]
            result = self.run_step(task, step)

            if self.need_replan(step, result):
                print("\n⚠️  触发重规划！")
                self.make_plan(task)
                continue

            self.current_step += 1

        print("\n🎉 全部任务完成！")

# ======================
# 运行（默认 DeepSeek，需在 .env 中配置 DEEPSEEK_API_KEY；改用 Qwen 则设置 LLM_PROVIDER=qwen）
# ======================
if __name__ == "__main__":
    agent = PlanReactToolAgent()
    agent.run("帮我查北京天气，再算 123+456，然后总结")