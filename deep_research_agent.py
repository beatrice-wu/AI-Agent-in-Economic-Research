"""
deep_research_agent.py: Deep Research Agent for Economic Research using LangGraph
================================================================================
A multi-agent system that performs comprehensive research on economic topics through:

1. Lead Researcher – Develops research strategy and synthesizes findings
2. Search Agents – Execute parallel searches across multiple sources
3. Analysis Agents – Analyze results from different perspectives
4. Synthesis Agent – Integrates all findings into a comprehensive report

Setup:
1. pip install langchain-openai langgraph tavily-python python-dotenv
2. Get API keys from:
   - OpenAI: https://platform.openai.com/api-keys
   - Tavily: https://app.tavily.com/
3. Create .env file with:
   OPENAI_API_KEY=...
   TAVILY_API_KEY=...
"""

import os
import json
import asyncio
from typing import Dict, List, TypedDict, Annotated, Literal
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from tavily import TavilyClient


# Load environment variables
load_dotenv()


# -----------------------------
# State definition
# -----------------------------
class ResearchState(TypedDict):
    question: str
    research_plan: Dict[str, List[str]]
    subtasks: List[Dict[str, str]]
    search_results: List[Dict[str, any]]
    analysis_results: List[Dict[str, str]]
    final_report: str
    messages: Annotated[List, add_messages]
    current_subtask: int
    max_iterations: int


# -----------------------------
# Deep Research Agent
# -----------------------------
class DeepResearchAgent:
    """Multi-agent system for deep economic research"""

    def __init__(self, openai_api_key: str = None, tavily_api_key: str = None):
        openai_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        tavily_key = tavily_api_key or os.getenv("TAVILY_API_KEY")

        if not openai_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY in .env")
        if not tavily_key:
            raise ValueError("Tavily API key not found. Set TAVILY_API_KEY in .env")

        # LLMs (you may change models if needed)
        self.lead_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            api_key=openai_key,
        )

        self.analysis_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            api_key=openai_key,
        )

        # Tools
        self.tavily = TavilyClient(api_key=tavily_key)
        self.executor = ThreadPoolExecutor(max_workers=10)

        # Build graph
        self.graph = self._build_graph()

    # -----------------------------
    # Graph construction
    # -----------------------------
    def _build_graph(self):
        workflow = StateGraph(ResearchState)

        workflow.add_node("lead_researcher", self.lead_researcher_node)
        workflow.add_node("spawn_subtasks", self.spawn_subtasks_node)
        workflow.add_node("search_agent", self.search_agent_node)
        workflow.add_node("analysis_agent", self.analysis_agent_node)
        workflow.add_node("synthesis_agent", self.synthesis_agent_node)

        workflow.set_entry_point("lead_researcher")
        workflow.add_edge("lead_researcher", "spawn_subtasks")
        workflow.add_edge("spawn_subtasks", "search_agent")
        workflow.add_conditional_edges(
            "search_agent",
            self.should_continue_searching,
            {
                "continue": "search_agent",
                "analyze": "analysis_agent",
            },
        )
        workflow.add_edge("analysis_agent", "synthesis_agent")
        workflow.add_edge("synthesis_agent", END)

        return workflow.compile()

    # -----------------------------
    # Nodes
    # -----------------------------
    def lead_researcher_node(self, state: ResearchState) -> ResearchState:
        print("Lead Researcher: Developing research strategy...")

        prompt = f"""
Develop a research strategy for: {state['question']}

Create 3–4 research directions and key questions.
Return JSON:
{{
  "research_directions": ["direction1", "direction2", "direction3"],
  "key_questions": ["question1", "question2", "question3"],
  "data_requirements": ["data1", "data2", "data3"]
}}
"""

        response = self.lead_llm.invoke([SystemMessage(content=prompt)])
        research_plan = json.loads(response.content)

        state["research_plan"] = research_plan
        state["messages"].append(
            HumanMessage(content=f"Research plan: {research_plan}")
        )

        print(
            f"Research strategy created with {len(research_plan['research_directions'])} directions"
        )
        return state

    def spawn_subtasks_node(self, state: ResearchState) -> ResearchState:
        print("Creating research subtasks...")

        subtasks = []
        directions = state["research_plan"]["research_directions"]
        questions = state["research_plan"]["key_questions"]

        for i, direction in enumerate(directions):
            if i < len(questions):
                subtasks.append(
                    {
                        "direction": direction,
                        "question": questions[i],
                        "status": "pending",
                        "search_queries": self._generate_search_queries(
                            direction, questions[i]
                        ),
                    }
                )

        if len(directions) >= 2 and len(questions) >= 2:
            subtasks.extend(
                [
                    {
                        "direction": directions[0],
                        "question": questions[1],
                        "status": "pending",
                        "search_queries": self._generate_search_queries(
                            directions[0], questions[1]
                        ),
                    },
                    {
                        "direction": directions[1],
                        "question": questions[0],
                        "status": "pending",
                        "search_queries": self._generate_search_queries(
                            directions[1], questions[0]
                        ),
                    },
                ]
            )

        state["subtasks"] = subtasks
        state["current_subtask"] = 0
        state["search_results"] = []
        state["analysis_results"] = []

        total_searches = sum(len(s["search_queries"]) for s in subtasks)
        print(f"Created {len(subtasks)} subtasks ({total_searches} searches)")

        return state

    def _generate_search_queries(self, direction: str, question: str) -> List[str]:
        prompt = f"""
Generate 2–3 search queries for:
Direction: {direction}
Question: {question}

Return JSON:
{{ "queries": ["query1", "query2", "query3"] }}
"""
        response = self.lead_llm.invoke([SystemMessage(content=prompt)])
        return json.loads(response.content)["queries"][:3]

    def search_agent_node(self, state: ResearchState) -> ResearchState:
        idx = state["current_subtask"]
        batch_size = 3
        end_idx = min(idx + batch_size, len(state["subtasks"]))

        print(f"Search Agent: processing subtasks {idx+1}–{end_idx}")

        futures = []
        for i in range(idx, end_idx):
            subtask = state["subtasks"][i]
            for query in subtask["search_queries"]:
                futures.append(
                    self.executor.submit(self._perform_search, query)
                )

        for future in futures:
            state["search_results"].extend(future.result())

        for i in range(idx, end_idx):
            state["subtasks"][i]["status"] = "searched"

        state["current_subtask"] = end_idx
        print(f"Collected {len(futures)} search results")

        return state

    def _perform_search(self, query: str) -> List[Dict]:
        try:
            results = self.tavily.search(
                query=query,
                search_depth="basic",
                max_results=3,
                include_answer=True,
                include_raw_content=False,
                include_images=False,
            )

            parsed = [
                {
                    "query": query,
                    "url": r.get("url", ""),
                    "title": r.get("title", ""),
                    "content": r.get("content", ""),
                    "score": r.get("score", 0),
                }
                for r in results.get("results", [])
            ]

            if results.get("answer"):
                parsed.insert(
                    0,
                    {
                        "query": query,
                        "url": "Tavily AI Summary",
                        "title": "AI-Generated Summary",
                        "content": results["answer"],
                        "score": 1.0,
                    },
                )

            return parsed

        except Exception as e:
            print(f"Search error for '{query}': {e}")
            return []

    def should_continue_searching(
        self, state: ResearchState
    ) -> Literal["continue", "analyze"]:
        if state["current_subtask"] < len(state["subtasks"]):
            return "continue"
        return "analyze"

    def analysis_agent_node(self, state: ResearchState) -> ResearchState:
        print("Analysis Agents: analyzing search results...")

        futures = []
        for subtask in state["subtasks"]:
            if subtask["status"] == "searched":
                relevant = [
                    r
                    for r in state["search_results"]
                    if r["query"] in subtask["search_queries"]
                ]
                futures.append(
                    self.executor.submit(
                        self._analyze_subtask_results, subtask, relevant
                    )
                )

        for subtask, future in zip(state["subtasks"], futures):
            state["analysis_results"].append(
                {"subtask": subtask, "analysis": future.result()}
            )

        print(f"Completed analysis for {len(state['analysis_results'])} subtasks")
        return state

    def _analyze_subtask_results(
        self, subtask: Dict, results: List[Dict]
    ) -> str:
        prompt = f"""
Analyze the following search results:

Direction: {subtask['direction']}
Question: {subtask['question']}

Results:
{json.dumps(results, indent=2)}

Provide a 300–500 word analysis covering:
- Key findings
- Evidence and data
- Different perspectives
- Practical implications

Cite sources as [Title, URL].
"""
        response = self.analysis_llm.invoke([SystemMessage(content=prompt)])
        return response.content

    def synthesis_agent_node(self, state: ResearchState) -> ResearchState:
        print("Synthesis Agent: creating final report...")

        prompt = f"""
Synthesize the following analyses into a comprehensive research report.

Question:
{state['question']}

Research Plan:
{json.dumps(state['research_plan'], indent=2)}

Analysis Results:
{json.dumps(state['analysis_results'], indent=2)}

Write a well-structured report (3–5 pages) including:
- Executive summary
- Key findings
- Cross-cutting insights
- Policy and research implications

Use [Source, Year] citations.
"""
        response = self.lead_llm.invoke([SystemMessage(content=prompt)])
        state["final_report"] = response.content

        print("Research report completed.")
        return state

    # -----------------------------
    # Execution helpers
    # -----------------------------
    async def research(self, question: str) -> str:
        initial_state: ResearchState = {
            "question": question,
            "research_plan": {},
            "subtasks": [],
            "search_results": [],
            "analysis_results": [],
            "final_report": "",
            "messages": [],
            "current_subtask": 0,
            "max_iterations": 10,
        }

        config = {"recursion_limit": 50}
        final_state = await self.graph.ainvoke(initial_state, config=config)

        return final_state["final_report"]

    def shutdown(self):
        self.executor.shutdown(wait=True)


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    agent = DeepResearchAgent()

    question = "What are the labor market effects of transformative AI expected to be?"

    print(f"Starting deep research on: {question}")
    print("-" * 80)

    report = asyncio.run(agent.research(question))

    print("\n" + "=" * 80)
    print("RESEARCH REPORT")
    print("=" * 80)
    print(report)

    agent.shutdown()
