from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from langchain.tools import DuckDuckGoSearchRun
import os
from dotenv import load_dotenv

load_dotenv()

# Use a strong + fast Groq model (Llama 3.3 70B is excellent in 2026)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.75,
    api_key=os.getenv("GROQ_API_KEY")
)

search_tool = DuckDuckGoSearchRun()

# === AGENTS ===
researcher = Agent(
    role="Senior AI Research Analyst",
    goal="Find the newest breakthroughs in recursive self-improvement, agent memory, and autonomous coding",
    backstory="You are obsessed with turning small toy projects into real seeds of the dingularity.",
    llm=llm,
    tools=[search_tool],
    verbose=True
)

coder = Agent(
    role="Elite Python Architect",
    goal="Write clean, safe code improvements to the Echo swarm itself",
    backstory="You output improvements as unified diffs that can be reviewed and applied.",
    llm=llm,
    verbose=True
)

evaluator = Agent(
    role="Ruthless Performance Judge",
    goal="Score every proposed change honestly on usefulness, safety, and dingularity potential",
    backstory="You are brutally honest but constructive.",
    llm=llm,
    verbose=True
)

improver = Agent(
    role="Recursive Self-Improvement Director",
    goal="Decide what gets kept and set the next goal for the swarm",
    backstory="You are building the future one loop at a time.",
    llm=llm,
    verbose=True
)

# === TASKS ===
research_task = Task(
    description="Search for the 3 most recent and interesting breakthroughs in self-improving agents, memory systems, or recursive AI from the last 45 days.",
    expected_output="Bullet list of 3 breakthroughs with short explanation and links if possible",
    agent=researcher
)

code_task = Task(
    description="Based on the research, propose specific improvements to echo_crew.py or run_echo.sh as a unified diff.",
    expected_output="A valid unified diff (```diff ... ```) that can be reviewed",
    agent=coder
)

eval_task = Task(
    description="Score the proposed code changes from 1-100 on usefulness, safety, simplicity, and dingularity potential. Explain your score.",
    expected_output="Score + detailed reasoning",
    agent=evaluator
)

improve_task = Task(
    description="Synthesize everything and give final verdict + clear next goal for the next loop.",
    expected_output="Final decision + next 24-hour goal for Echo",
    agent=improver
)

# === CREW ===
echo_crew = Crew(
    agents=[researcher, coder, evaluator, improver],
    tasks=[research_task, code_task, eval_task, improve_task],
    process=Process.sequential,
    memory=True,
    verbose=2
)

def run_echo():
    print("🔄 === ECHO SELF-IMPROVEMENT LOOP STARTED ===\n")
    result = echo_crew.kickoff()
    print("\n" + "="*60)
    print("ECHO OUTPUT:")
    print(result)
    print("="*60)
    return result

if __name__ == "__main__":
    run_echo()
