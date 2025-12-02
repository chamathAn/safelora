from xml.parsers.expat import model
from unsloth import FastLanguageModel
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
        prompt = """
You are the Safelora Final Advisory Agent.

You receive a structured technical summary about a plant disease case.  
Use this summary to write a clear, descriptive, and practical advisory for a farmer or field officer.  
Your output must be written mainly in natural paragraphs, with steps included where needed, but avoid short fragmented bullet lists unless necessary for clarity.

Do not add any chemicals, products, or practices that are not explicitly supported by the summary.  
Do not mention internal agents, pipelines, or tools.  
Focus only on the information provided.

Follow this output structure exactly, but write each section in flowing, descriptive paragraphs:

1. **Case Summary**  
   Describe the situation in a short paragraph, restating the crop, disease, crop stage, and weather conditions.

2. **Disease Explanation**  
   Explain the disease in a farmer-friendly descriptive way. Include what causes it, how it affects the plant, and how symptoms typically develop over time.

3. **Risk Assessment**  
   Provide a simple risk level (Low / Moderate / High) and explain in a descriptive paragraph why the risk is at that level based on stage, weather, and disease behavior.

4. **Immediate Actions (0–7 Days)**  
   Give clear, practical steps the farmer should take right now.  
   Write in descriptive paragraphs and include step-by-step actions when needed.  
   If the summary includes safe chemical options, mention them without adding anything new.

5. **Short-Term Management (2–4 Weeks)**  
   Explain the follow-up actions the farmer should continue over the next few weeks.  
   Describe what to monitor, how often, and why timing matters for the crop stage and weather.

6. **Long-Term Prevention**  
   Describe preventive approaches for future seasons in paragraph form, using only what is supported by the summary (e.g., rotation, sanitation, spacing, resistant varieties if mentioned).

7. **Stage- and Weather-Specific Notes**  
   Write a descriptive paragraph summarizing how the current crop stage and weather influence disease behavior, progress, and recommended actions.

8. **Safety and Handling Notes**  
   Provide a short explanatory paragraph on safety, PPE, and safe behavior, especially if chemicals are included in the summary.

9. **When to Seek Further Help**  
   Explain clearly when and why the farmer should contact an agricultural officer or advisor if symptoms worsen or do not match expectations.

Write everything in smooth, descriptive paragraphs with clear reasoning.  
Do not mention that you are an AI.  
Do not repeat the instructions.

--------------------
TECHNICAL SUMMARY:
{context_summary}
--------------------

Generate the advisory now.
"""

        max_seq_length = 6000
        dtype = None
        load_in_4bit = True
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        ADAPTER_DIR = os.path.join(BASE_DIR, "LLM", "safefelora_lora_adapters")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=ADAPTER_DIR,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            device_map={"": 0},
        )
        context_input = getattr(self, "context_agent_content", "") or ""
        encoded = tokenizer(
            [prompt.format(context_summary=context_input)],
            return_tensors="pt",
        ).to(model.device)
        FastLanguageModel.for_inference(model)
        outputs = model.generate(
            **encoded,
            max_new_tokens=1000,
            temperature=0.7,
            top_p=0.9,
        )

        result = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        print("=" * 80 + "\nGenerative agent prompt\n" + "=" * 80)
        print(prompt.format(context_summary=context_input))
        print("-" * 80 + "\nGenerative agent result:\n" + "-" * 80)
        print(result)
        print("=" * 162)
        return {"messages": [AIMessage(content=result)]}


__all__ = ["OrchestrateAgent"]
