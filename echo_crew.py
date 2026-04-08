from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from langchain.tools import DuckDuckGoSearchRun
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY")
)

search_tool = DuckDuckGoSearchRun()

# Agents
researcher = Agent(
    role="Senior AI Research Analyst",
    goal="Identify recent breakthroughs in self-improving agents",
    backstory="Expert at finding relevant research.",
    llm=llm,
    tools=[search_tool],
    verbose=True
)

coder = Agent(
    role="Senior Python Engineer",
    goal="Propose clean code improvements",
    backstory="Creates maintainable modifications.",
    llm=llm,
    verbose=True
)

evaluator = Agent(
    role="System Evaluator",
    goal="Provide honest assessment of changes",
    backstory="Evaluates practicality and safety.",
    llm=llm,
    verbose=True
)

improver = Agent(
    role="Improvement Director",
    goal="Make final decisions on changes",
    backstory="Coordinates the improvement process.",
    llm=llm,
    verbose=True
)

hyperreflector = Agent(
    role="HyperReflector",
    goal="Extract meta-lessons from the loop",
    backstory="Learns from the improvement process itself.",
    llm=llm,
    verbose=True
)

# Tasks
research_task = Task(
    description="Find and summarize the 3 most recent breakthroughs on self-improving agents from the last 60 days.",
    expected_output="3 bullet points with sources.",
    agent=researcher
)

code_task = Task(
    description="Propose specific improvements to echo_crew.py as a unified diff.",
    expected_output="Valid unified diff in code block.",
    agent=coder
)

eval_task = Task(
    description="Score the proposed changes 1-100 and explain.",
    expected_output="Score with reasoning.",
    agent=evaluator
)

improve_task = Task(
    description="Give final recommendation and next goal.",
    expected_output="Decision and next goal.",
    agent=improver
)

reflect_task = Task(
    description="Review this loop and extract 3-5 actionable meta-lessons.",
    expected_output="Bullet list of meta-lessons.",
    agent=hyperreflector
)

# Crew
echo_crew = Crew(
    agents=[researcher, coder, evaluator, improver, hyperreflector],
    tasks=[research_task, code_task, eval_task, improve_task, reflect_task],
    process=Process.sequential,
    memory=True,
    verbose=2
)

def run_echo():
    print("Starting Echo Swarm v0.2\n")
    result = echo_crew.kickoff()
    print("\n" + "="*80)
    print("LOOP RESULT:")
    print(result)
    print("="*80)

if __name__ == "__main__":
    run_echo()
