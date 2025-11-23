from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain.tools import tool
from langchain_core.messages import AnyMessage, ToolMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma

from pydantic import BaseModel, Field
from unsloth import FastLanguageModel
from dotenv import load_dotenv
import os

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# tools
search_web = TavilySearch(max_results=2)


@tool
def tavily_search(query: str) -> str:
    """Search the web for information using Tavily."""
    return search_web.invoke(query)


@tool
def search_local_knowledge_base(query: str) -> str:
    """Search the local knowledge base for plant disease information."""
    embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-en")
    vectorstore = Chroma(
        persist_directory="../kb_db",
        embedding_function=embeddings,
    )
    retriever = vectorstore.as_retriever(k=5)
    docs = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs)


tools = [tavily_search, search_local_knowledge_base]


# safety response
class SafetyResponse(BaseModel):
    is_safe: bool = Field(..., description="Indicates if the content is safe.")
    reasons: list[str] = Field(
        ..., description="List of reasons explaining the safety assessment."
    )


class OrchestrateAgentState(TypedDict):
    plant: str
    disease: str
    crop_stage: str
    weather: str
    is_safe: bool
    messages: Annotated[list[AnyMessage], operator.add]


class OrchestrateAgent:
    def __init__(self, checkpointer):
        self.llm = ChatGroq(model="qwen/qwen3-32b", temperature=0)

        self.tools = {t.name: t for t in tools}
        self.llm_with_tools = self.llm.bind_tools(tools)
        self.llm_safety = self.llm.with_structured_output(
            SafetyResponse, method="json_mode"
        )

        workflow = StateGraph(OrchestrateAgentState)
        workflow.add_node("context_agent", self.context_node)
        workflow.add_node("action", self.tool_executor)
        workflow.add_node("safety_agent", self.safety_node)
        workflow.add_node("generative_agent", self.generative_node)

        workflow.add_conditional_edges(
            "context_agent",
            self.exists_tool_action,
            {True: "action", False: "safety_agent"},
        )

        workflow.add_conditional_edges(
            "safety_agent",
            self.exists_safety,
            {True: "generative_agent", False: "context_agent"},
        )

        workflow.add_edge("action", "context_agent")
        workflow.add_edge("generative_agent", END)
        workflow.set_entry_point("context_agent")

        self.graph = workflow.compile(checkpointer=checkpointer)

    # ============================================================================= context agent ======================================================================

    def context_node(self, state: OrchestrateAgentState):
        print("=" * 80 + "\nContext agent is running....\n" + "=" * 80)
        messages = state["messages"][-3:]
        plant = state["plant"]
        disease = state["disease"]
        crop_stage = state["crop_stage"]
        weather = state["weather"]

        final_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are the Safelora Context Retrieval Agent.

Your goal is to gather accurate, factual plant-disease information and produce a
structured summary (no recommendations, no invented facts).

You must also comply with safety requirements:
- Previous messages may contain Safety Agent feedback explaining why earlier output was unsafe.
- You MUST read that feedback carefully.
- You MUST ensure that your new output avoids all unsafe elements flagged earlier.
- If the Safety Agent indicated missing or unsafe chemical information, you must:
    • remove or avoid unsafe chemicals,
    • replace them with safe factual alternatives ONLY if supported by sources,
    • fill gaps using limited tool calls,
    • produce content that is expected to pass the Safety Agent.
- Never reintroduce banned or restricted pesticides, unsafe mixing instructions,
  overdose behavior, or anything the Safety Agent marked unsafe.

Use a compact ReAct workflow:

1. REASON  
   - Based on the plant, disease, crop stage, and current weather, list the key
     information needed for the final summary.
   - Identify which parts may require stage-specific or weather-specific detail.
   - Identify which parts must be corrected or replaced due to Safety Agent feedback.

2. ACT — with controlled tool usage  
   You have two tools:

   • `search_local_knowledge_base(query)`  
     - Primary source (PlantwisePlus + UC IPM).  
     - Build ONE well-planned query that combines:  
       - plant  
       - disease  
       - crop stage  
       - current weather  
       - and all required information categories (symptoms, pathogen, spread,
         environmental factors, management, prevention).  
     - Extract as much as possible from this single call and verify that it aligns
       with safety requirements.

   • `tavily_search(query)`  
     - Secondary source for confirmation and gaps.  
     - After using the local knowledge base, identify the most important missing,
       uncertain, or safety-related points flagged by the Safety Agent.  
     - Build ONE focused `tavily_search` query that:
         - fills those gaps and/or
         - cross-checks critical facts from the local knowledge,
         - ensures the final content complies with safety requirements.
     - You should call `tavily_search` at least once, but only once, in this process.

   Do not run unnecessary loops. The flow should be:
   → one local KB call  
   → one focused Tavily call for missing/unsafe parts  
   → then stop.

3. OBSERVE  
   - Read tool results carefully and extract only factual, supported information.
   - Prefer information consistent with earlier safety feedback.

4. DECIDE  
   - When enough information is gathered (or clear gaps remain), stop tool usage
     and generate the structured summary.
   - Ensure your output complies with all safety guidance and contains no unsafe elements.

---

### Final Output Structure

1. Disease Overview  
2. Symptoms and Signs  
3. Causative Agent  
4. Transmission and Spread  
5. Environmental & Crop Stage Factors  
6. Management Information (raw, source-based only)  
7. Prevention Measures  
8. Stage-Specific Notes  
9. Weather-Specific Notes  
10. Sources Used (local KB, tavily, or both)

Guidelines:
- Do NOT include recommendations.  
- Do NOT invent missing information.  
- Do NOT include unsafe pesticide information or anything flagged previously.  
- Mark missing sections clearly instead of calling more tools.  
- Do NOT mention tools, RAG, or system internals in the final output.  
- Keep the writing technical, clean, and concise.  
""",
                ),
                (
                    "human",
                    """
Plant: {plant}
Disease: {disease}
Crop Stage: {crop_stage}
Weather: {weather}

Messages so far:
{messages}
""",
                ),
            ]
        )

        formatted_prompt = final_prompt.format_prompt(
            plant=plant,
            disease=disease,
            crop_stage=crop_stage,
            weather=weather,
            messages=messages,
        )

        response = self.llm_with_tools.invoke(formatted_prompt)
        if hasattr(response, "content"):
            self.context_agent_content = str(response.content)
        else:
            self.context_agent_content = None
        return {"messages": [response]}

    def exists_tool_action(self, state):
        result = state["messages"][-1]
        return len(result.tool_calls) > 0

    def tool_executor(self, state: OrchestrateAgentState):
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for t in tool_calls:
            print("=" * 80 + f"\n Calling {t['name']}....\n" + "=" * 80)
            result = self.tools[t["name"]].invoke(t["args"])
            results.append(
                ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(result))
            )
        print("Back to the model!")
        return {"messages": results}

    # =========================================================================== safety agent ============================================================================

    def safety_node(self, state: OrchestrateAgentState):
        print("=" * 80 + "\nSafety agent is running....\n" + "=" * 80)

        safety_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are the Safelora Safety Agent.

Your job is to strictly evaluate plant disease management CONTENT for safety,
using the SAFETY_CONTEXT as the authoritative reference. The SAFETY_CONTEXT
contains validated information from sources such as WHO pesticide classifications
and the Sri Lankan Department of Agriculture.

You do NOT rewrite the content.
You ONLY judge:
- whether the CONTENT is safe or not, and
- why you made that decision,
- and, if it is unsafe, what would need to change for it to become safe.

SAFETY-FIRST RULES (STRICT BUT NOT EXTREME):

1. BANNED / RESTRICTED / HIGHLY HAZARDOUS CHEMICALS
   - If the CONTENT mentions any pesticide or chemical that appears in the SAFETY_CONTEXT
     as banned, severely restricted, or highly hazardous → is_safe = false.

2. CLEARLY UNAPPROVED OR RISKY CHEMICAL USE
   - If the CONTENT describes a pesticide use that clearly conflicts with SAFETY_CONTEXT
     (e.g. used on a crop/disease where it is explicitly not allowed or warned against) →
     is_safe = false.
   - If the CONTENT strongly suggests misuse (e.g. obviously excessive rates, very frequent
     applications without justification) → is_safe = false.

3. MISSING SAFETY DETAILS
   - Missing details (e.g. no dose or PPE mentioned) do NOT automatically make the CONTENT unsafe,
     but if the omission creates a realistic risk of misuse or contradicts SAFETY_CONTEXT guidance,
     you should treat it as unsafe and explain why.
   - If the chemical use is generic and not clearly dangerous, you may still consider the CONTENT
     safe while noting that details are not fully specified.

4. CONTRADICTIONS WITH SAFETY_CONTEXT
   - If the CONTENT directly contradicts clear safety statements in SAFETY_CONTEXT
     (e.g. SAFETY_CONTEXT warns against a product or method that CONTENT promotes) →
     is_safe = false.

5. CLEARLY UNSAFE PRACTICES
   - If the CONTENT promotes behaviour such as:
     - obvious overdose or very frequent spraying,
     - unsafe tank mixes or “mix many chemicals together” without any justification,
     - clearly ignoring label/safety instructions,
     - spraying near water sources, food, children, or without any basic protection,
     then → is_safe = false.

6. AMBIGUITY RULE (MODERATE)
   - If you cannot confidently determine safety, describe the uncertainty in `reasons`.
   - Only set is_safe = false in ambiguous cases when the ambiguity itself presents
     a realistic safety concern or potential for harm.
   - If ambiguity is minor and there is no clear indication of risk, you may still set
     is_safe = true and explain why.

7. WHEN CAN CONTENT BE MARKED SAFE?
   - Mark is_safe = true when ALL of the following are true:
     - The CONTENT does NOT mention any banned, restricted, or highly hazardous chemical.
     - There is no clear evidence of misuse or direct conflict with SAFETY_CONTEXT.
     - There is no obviously reckless or dangerous practice described.
   - Purely cultural or biological practices that align with general safety principles
     should normally be considered safe.

Required output format:
You MUST return ONLY a JSON object in this exact structure:

{{
  "is_safe": <true_or_false>,
  "reasons": ["short explanation 1", "short explanation 2"]
}}

Behavior of `reasons` based on is_safe:

- If is_safe = false:
  - `reasons` MUST include:
    - at least one short sentence explaining WHAT is unsafe
      (e.g. "Unsafe because it mentions banned pesticide X.").
    - at least one short sentence explaining WHAT SHOULD CHANGE so the content could be considered safe
      (e.g. "To make this content safe, remove pesticide X and replace it with a permitted option according to SAFETY_CONTEXT."
             or "To make this content safe, adjust the advice to follow the recommended rate and interval given in SAFETY_CONTEXT.").
  - These are NOT full agronomic recommendations, only safety-oriented corrections.

- If is_safe = true:
  - `reasons` should briefly explain WHY it is considered safe
    (e.g. "Only cultural practices are mentioned and no chemicals are used.",
           "No banned or restricted pesticides appear and the advice does not conflict with SAFETY_CONTEXT.").
  - Do NOT suggest changes if it is already safe.

Strict Output Rules:
- `is_safe` must be true or false.
- `reasons` must be a list of short, clear sentences.
- Do NOT output anything before or after the JSON.
- Do NOT add extra fields.
- Do NOT restate the full content—only justify the safety judgement and, if unsafe, how to make it safe.

Be careful, safety-focused, and fair.
Flag clear risks as unsafe, but do not automatically treat every uncertainty as unsafe.
""",
                ),
                (
                    "human",
                    """
CONTENT TO EVALUATE:
{text}

SAFETY_CONTEXT:
{context}

Evaluate the CONTENT using the SAFETY_CONTEXT and return ONLY the structured JSON safety assessment.
""",
                ),
            ]
        )

        message = state["messages"][-1]

        embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-en")
        vectorstore = Chroma(
            persist_directory="../safety_kb_db",
            embedding_function=embeddings,
        )
        retriever = vectorstore.as_retriever(k=5)

        safety_chain = (
            {
                "text": RunnablePassthrough(),
                "context": retriever
                | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
            }
            | safety_prompt
            | self.llm_safety
        )
        safety_response = safety_chain.invoke(str(message.content))

        return {
            "is_safe": safety_response.is_safe,
            "messages": [AIMessage(content=str(safety_response))],
        }

    def exists_safety(self, state):
        return state["is_safe"]

    # ================================================ Generative Agent ========================================================
    def generative_node(self, state: OrchestrateAgentState):
        alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are the Safelora Final Advisory Agent.

You receive as INPUT a structured technical context summary about a specific plant disease case.
This summary is produced by an internal context agent and already includes:
- Plant, disease, crop stage, and current weather
- Disease overview, symptoms, pathogen, spread
- Environmental and crop-stage factors
- Management and prevention information
- Stage-specific and weather-specific notes
- Safety-screened information (unsafe elements should already be minimized, but you must still be cautious)

Your job is to transform this technical context into a clear, actionable, and safe advisory
for a farmer or field officer.

You MUST follow these rules:

1. General role and scope
   - Provide a practical, field-ready advisory.
   - Focus on helping the farmer manage the current situation and reduce future risk.
   - Base ALL advice ONLY on information contained or clearly implied in the INPUT.
   - Do NOT invent new chemicals, products, or practices that are not supported by the INPUT.
   - Do NOT mention internal systems, agents, RAG, tools, or any data sources.

2. Safety and compliance
   - Assume that banned or restricted pesticides must NOT be recommended.
   - If the INPUT clearly includes safe and permitted chemical options, you may use them,
     but do not add new chemicals by name.
   - If chemical options are vague or incomplete in the INPUT, you may say that the farmer
     should consult local agricultural authorities or label guidance for specific products,
     instead of guessing.
   - Always emphasize safe handling, PPE, and careful use when chemicals are mentioned,
     but keep it concise.
   - If the INPUT only contains cultural or biological methods, focus on those and avoid
     introducing chemicals.

3. Use plant, disease, crop stage, and weather
   - Always consider the specific:
     - plant (crop),
     - disease,
     - crop growth stage, and
     - current weather conditions
     described or implied in the INPUT.
   - Make your advice stage-aware (e.g., seedling vs. flowering vs. fruiting)
     and weather-aware (e.g., wet/humid vs. dry/hot).

4. Required output structure
   Your RESPONSE must follow this structure, with headings in this exact order:

   1. Case Summary
      - Briefly restate the situation:
        crop, disease, crop stage, and current/recent weather.
      - Mention diagnosis confidence if clearly implied (e.g., “Symptoms strongly match X”).

   2. Disease Explanation (Farmer-Friendly)
      - Explain in simple terms what the disease is and how it affects the crop.
      - Describe key visible symptoms in a way a farmer can recognize.
      - Mention how the disease usually develops over time.

   3. Risk Assessment
      - Give a short qualitative risk level based on the INPUT (e.g., “Low”, “Moderate”, “High”).
      - Explain WHY (e.g., favorable weather, crop stage sensitivity, severity of symptoms).
      - If relevant, mention risk of yield loss or spread to nearby plants/fields.

   4. Immediate Actions (0–7 Days)
      - List concise, practical actions the farmer can start right away.
      - Organize by type where applicable:
        - Cultural / field hygiene (e.g., roguing, sanitation, irrigation adjustments)
        - Biological / natural approaches (if mentioned or implied)
        - Chemical actions ONLY if clearly supported by INPUT and safe:
          - Do NOT invent product names, only refer to those present in the INPUT.
          - Do NOT guess rates or schedules beyond what is supported.
      - Keep each action as a clear bullet point.

   5. Short-Term Management (Next 2–4 Weeks)
      - Describe follow-up actions to keep the disease under control.
      - Include monitoring frequency, what symptoms to watch for, and any timing linked
        to crop stage or weather.
      - Maintain the same type grouping (Cultural / Biological / Chemical) when useful.

   6. Long-Term Prevention
      - Based on the INPUT, describe preventive measures for future seasons:
        - Resistant varieties (if mentioned)
        - Crop rotation or field history considerations
        - Seed/planting material health
        - Canopy management, spacing, irrigation, or other long-term practices
      - Do NOT invent new resistant varieties or products if they are not mentioned.

   7. Stage- and Weather-Specific Notes
      - Clearly summarize any advice that is specific to:
        - the current crop stage, and/or
        - the current or forecasted weather conditions.
      - If the INPUT states that certain conditions increase or decrease risk, explain it simply.

   8. Safety and Handling Notes
      - Summarize key safety points relevant to the actions you listed:
        - PPE and safe handling if chemicals are mentioned in the INPUT.
        - Avoiding contamination of water, food, children, and animals.
      - Keep this section short but explicit.
      - If INPUT contains no chemicals at all, focus only on general hygiene and safe practices.

   9. When to Seek Further Help
      - Briefly describe situations where the farmer should seek additional help:
        - If symptoms do not match the description,
        - If disease continues to worsen despite actions taken,
        - If local regulations or product labels provide additional constraints.
      - Encourage consulting local agricultural officers, extension agents, or certified advisors,
        without naming any specific organization unless clearly implied in the INPUT.

5. Style and tone
   - Be clear, structured, and practical.
   - Use short paragraphs and bullet points where appropriate.
   - Avoid technical jargon where a simpler term exists.
   - Write as if speaking to a reasonably experienced farmer or field officer.
   - Do NOT mention that you are an AI or part of Safelora; just provide the advisory.

Your task: Using ONLY the information given in the INPUT, generate the final advisory
in the exact structure described above.

### Input:
{input}

### Response:
"""

        max_seq_length = 6000
        dtype = None
        load_in_4bit = True
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="./LLM/safefelora_lora_adapters",
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
        FastLanguageModel.for_inference(model)
        inputs = tokenizer(
            [
                alpaca_prompt.format(
                    input=(
                        self.context_agent_content
                        if hasattr(self, "context_agent_content")
                        else ""
                    ),
                    response="",
                )
            ],
            return_tensors="pt",
        ).to("cuda")
        outputs = model.generate(
            **inputs, max_new_tokens=1000, temperature=0.7, top_p=0.9
        )
        print(tokenizer.batch_decode(outputs, skip_special_tokens=True))


__all__ = ["OrchestrateAgent"]
