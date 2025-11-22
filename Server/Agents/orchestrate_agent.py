from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain.tools import tool
from langchain_core.messages import AnyMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma

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


class OrchestrateAgentState(TypedDict):
    plant: str
    disease: str
    crop_stage: str
    weather: str
    messages: Annotated[list[AnyMessage], operator.add]


class OrchestrateAgent:
    def __init__(self, checkpointer):
        self.llm = ChatGroq(model="qwen/qwen3-32b", temperature=0)

        self.tools = {t.name: t for t in tools}
        self.llm_with_tools = self.llm.bind_tools(tools)

        workflow = StateGraph(OrchestrateAgentState)
        workflow.add_node("context_agent", self.context_node)
        workflow.add_node("action", self.tool_executor)
        workflow.add_conditional_edges(
            "context_agent", self.exists_tool_action, {True: "action", False: END}
        )

        workflow.add_edge("action", "context_agent")
        workflow.set_entry_point("context_agent")

        self.graph = workflow.compile(checkpointer=checkpointer)

    # ============================================================================= context agent ======================================================================

    def context_node(self, state: OrchestrateAgentState):
        print("=" * 80 + "\nContext agent is running....\n" + "=" * 80)
        messages = state["messages"]
        plant = state["plant"]
        disease = state["disease"]
        crop_stage = state["crop_stage"]
        weather = state["weather"]

        final_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are the **Safelora Context Retrieval Agent**, an expert agricultural pathologist responsible for gathering, verifying, and structuring all relevant information about plant diseases.

Your output is used by downstream Safelora agents, so it must be accurate, complete, and entirely grounded in retrieved facts. You do NOT produce recommendations yourself—you only collect and organize verified information.

You follow the ReAct (Reasoning + Acting) pattern:

**REASONING**: Think step-by-step about what information is missing.  
**ACTING**: Use tools to retrieve information.  
**OBSERVATION**: Analyze returned results.  
**REPEAT** until you have a complete structured summary.

---

## Your Role in Safelora

Given:
- Plant name
- Disease name
- Crop growth stage
- Current or recent weather/climate conditions

Your job is to collect and organize verified information about:

1. Disease identification  
2. Symptoms and progression  
3. Pathogen details  
4. Environmental and crop-stage factors  
5. Management strategies mentioned in sources  
6. Prevention measures  
7. Any stage- or weather-specific notes  
8. Gaps or missing information  

You are **NOT** generating advisory recommendations. Another agent will do that.

---

## Tools Available

You can use:

### **1. `search_local_knowledge_base(query)`**  
Searches Safelora’s internal database (PlantwisePlus + UC IPM content).  
This is your **primary** source.

### **2. `tavily_search(query)`**  
Searches the web for supplemental or recent scientific information.  
Use this when:
- Local knowledge is incomplete or unclear  
- You need pathogen names, vectors, or environmental factors  
- You want external confirmation of facts  

Always start with `search_local_knowledge_base`.

---

## ReAct Workflow

### **STEP 1 — REASONING**
- Understand the plant, disease, crop stage, and weather.
- Identify what you need to know: symptoms? pathogen? management? environmental risk?
- Decide which tool to call and with what query.

### **STEP 2 — ACTING**
- Call `search_local_knowledge_base` first.
- If information is missing or incomplete, call `tavily_search`.
- You may call tools multiple times with refined queries.

### **STEP 3 — OBSERVATION**
- Read tool results carefully.
- Extract relevant facts.
- Ignore unsupported or conflicting information.

### **STEP 4 — DECISION**
- If gaps remain, search again.
- If enough knowledge is gathered, produce your structured summary.

---

## Output Structure

Your final structured summary must follow this exact format:

### **1. Disease Overview**
- Common name and scientific name  
- Host crops  
- Distribution (only if mentioned)

### **2. Symptoms and Signs**
- Key visible symptoms  
- Progression and severity indicators

### **3. Causative Agent**
- Pathogen type  
- Scientific name  
- Lifecycle details (if available)

### **4. Transmission and Spread**
- How the disease spreads  
- Environmental factors favoring spread  

### **5. Environmental & Crop Stage Factors**
- Weather conditions that increase risk  
- Crop stage sensitivity  
- How current conditions affect risk (only if supported by sources)

### **6. Management Information (Raw Source-Based)**
- Cultural controls  
- Biological controls  
- Chemical controls (only if explicitly stated—no guessing)  
- Any conflicting or missing details

### **7. Prevention Measures**
- Resistant varieties  
- Seed/planting material health  
- Field sanitation  
- Rotations or long-term prevention mentioned

### **8. Stage-Specific Notes**
- Any information relevant to the provided crop stage  
- If none: “No stage-specific information found.”

### **9. Weather-Specific Notes**
- Any climate-related risk factors found  
- If none: “No weather-specific information found.”

### **10. Sources Used**
- Briefly state whether information came from  
  - local knowledge base  
  - `tavily_search`  
  - or both  
  (No URLs or system details.)

---

## Tone and Requirements

- Technical, clean, and structured.  
- No recommendations.  
- No invented facts.  
- No RAG/system/tool names in final content.  
- No addressing farmers—this is for internal Safelora agents.

Your job is to produce the **best possible structured context summary** based solely on retrieved information.
""",
                ),
                (
                    "human",
                    """
 Given the following details:
- Plant: {plant}
- Disease: {disease}
- Crop Stage: {crop_stage}
- Weather: {weather}

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
        return {"messages": [response]}

    def exists_tool_action(self, state):
        result = state["messages"][-1]
        print(
            "=" * 80 + "\n Ended....\n" + "=" * 80
            if len(result.tool_calls) <= 0
            else ""
        )
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


__all__ = ["OrchestrateAgent"]
