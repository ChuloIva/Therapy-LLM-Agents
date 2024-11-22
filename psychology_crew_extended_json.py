from crewai import Agent, Task, Crew, Process
import openai
from pydantic import BaseModel
from typing import List
from langchain_ollama import ChatOllama
import os
os.environ["OPENAI_API_KEY"] = "NA"

# Define your LLM model (assuming ChatOllama is properly configured)
llm = ChatOllama(
    model="llama3.1:latest",
    base_url="http://localhost:11434"
)


# Function to read text from a file
def read_file_content(file_path):
    with open(file_path, 'r') as file:
        return file.read()

file_path_task1 = '/home/koalacrown/Desktop/Code/Projects/CR/Projects/therapy transcript'

class ComprehensiveSummaryReportModel(BaseModel):
    general_topics: List[str]
    cognitive_dissonances: List[str]
    emotional_traumas: List[str]
    long_term_healing_plan: List[str]
    final_action_plan: List[str]
    timeline: List[str]
    resources: List[str]



# Creating agents
cognitive_analyst = Agent(
    role='Cognitive Dissonance/Bias Analyst',
    goal='Analyze cognitive dissonances or biases in the client’s text',
    backstory='Skilled at detecting mental conflicts or biases within a conversation or written material.',
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

emotional_analyst = Agent(
    role='Emotional Trauma Analyst',
    goal='Analyze emotional trauma or core psychological issues in the client’s text',
    backstory='Expert in analyzing emotional content and identifying underlying psychological issues.',
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

topic_segmenter = Agent(
    role='Topic Segmenter',
    goal='Organize the client text into distinct general topics for further analysis',
    backstory='Adept at categorizing information into coherent topics for easier analysis by other agents.',
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

plan_generator = Agent(
    role='Therapy Plan Generator',
    goal='Generate actionable advice and a forward plan for the client',
    backstory='Focuses on creating concrete, practical advice to help clients move forward.',
    llm=llm,
    verbose=True,
    allow_delegation=False,
)


#Tasks
segment_task = Task(
    description=(
        'Segment the client’s input into distinct general topics. '
        'Each topic should represent a key issue or theme within the client’s input. '
        'Output the result as a JSON file and avoid any non-JSON content.'
        + "\n\nAdditional Info:\n" 
        + read_file_content(file_path_task1)
    ),
    expected_output=(
        'A structured JSON array of general topics extracted from the client’s text. '
        'For example:\n'
        '{\n'
        '  "topics": [\n'
        '    {"topic": "Relationship issues", "quote": "Example quote from the text"},\n'
        '    {"topic": "Work-related stress", "quote": "Example quote from the text"},\n'
        '    {"topic": "Anxiety about future decisions", "quote": "Example quote from the text"}\n'
        '  ]\n'
        '}'
    ),
    agent=topic_segmenter,
    # output_json=True,
    output_file='Topics.json',
)

cognitive_task = Task(
    description=(
        'Analyze cognitive dissonances or biases in the segmented text. '
        'Identify any contradictions or biases affecting the client’s decision-making or thought process. '
        'Output the result as a JSON file and avoid any non-JSON content.'
        + "\n\nAdditional Info:\n" 
        + read_file_content(file_path_task1)
    ),
    expected_output=(
        'A JSON array listing cognitive dissonances or biases. For example:\n'
        '{\n'
        '  "cognitive_dissonances": [\n'
        '    {"contradiction": "Client expresses desire for change but fears it simultaneously", '
        '     "quote": "Example quote from the text", '
        '     "cause": "Fear of failure"}\n'
        '  ]\n'
        '}'
    ),
    agent=cognitive_analyst,
    # output_json=True,
    output_file='Cognitive-biases.json',
)

emotional_task = Task(
    description=(
        'Analyze the client’s text for emotional trauma or core psychological issues. '
        'Identify past experiences or recurring patterns contributing to the client’s current situation. '
        'Output the result as a JSON file and avoid any non-JSON content.'
        + "\n\nAdditional Info:\n" 
        + read_file_content(file_path_task1)
    ),
    expected_output=(
        'A JSON array highlighting emotional traumas or core issues. For example:\n'
        '{\n'
        '  "emotional_issues": [\n'
        '    {"issue": "Fear of abandonment", '
        '     "quote": "Example quote from the text", '
        '     "impact": "The client struggles with trust in relationships."}\n'
        '  ]\n'
        '}'
    ),
    agent=emotional_analyst,
    # output_json=True,
    output_file='Emotional_analysis.json',
)

cognitive_plan_task = Task(
    description=(
        'Generate an actionable plan to help the client deal with cognitive dissonances or biases. '
        'Output the result as a JSON file and avoid any non-JSON content.'
    ),
    expected_output=(
        'A JSON array of concrete steps or strategies for cognitive dissonances. For example:\n'
        '{\n'
        '  "cognitive_plan": [\n'
        '    {"dissonance": "Fear of failure", '
        '     "strategy": "Challenge the belief by reflecting on past successes", '
        '     "resources": "Relevant articles, self-reflection techniques"}\n'
        '  ]\n'
        '}'
    ),
    agent=plan_generator,
    Context=[cognitive_task],  # Depends on the cognitive dissonance analysis
    # output_json=True,
    output_file='/home/koalacrown/Desktop/Code/Projects/CR/Projects/Psychology_crew/Data/json/Cognitive_resolvement.json',
)

forward_plan_task = Task(
    description=(
        'Create a forward-looking plan based on the emotional issues uncovered. '
        'Output the result as a JSON file and avoid any non-JSON content.'
    ),
    expected_output=(
        'A JSON array of actionable steps for emotional healing. For example:\n'
        '{\n'
        '  "emotional_plan": [\n'
        '    {"issue": "Fear of abandonment", '
        '     "steps": "Encourage building trust in relationships gradually", '
        '     "therapeutic_exercises": "Journaling, mindfulness practices"}\n'
        '  ]\n'
        '}'
    ),
    agent=plan_generator,
    Context=[emotional_task],  # Depends on the emotional analysis
    # output_json=True,
    output_file='Forward_plan.json',
)

summary_task = Task(
    description=(
        'Compile all outputs into a final summary for the client. '
        'This should provide a holistic view of the client’s situation, highlighting key cognitive and emotional issues, and proposed solutions. '
        'Output the result as a JSON file and avoid any non-JSON content.'
    ),
    expected_output=(
        'A comprehensive JSON report. For example:\n'
        '{\n'
        '  "summary": {\n'
        '    "topics": [...],\n'
        '    "cognitive_dissonances": [...],\n'
        '    "emotional_issues": [...],\n'
        '    "action_plan": {...}\n'
        '  }\n'
        '}'
    ),
    agent=plan_generator,
    Context=[cognitive_plan_task, forward_plan_task],  # Summarizes all previous tasks
    # output_json=True,
    output_file='Full_summary.json',
)


# Creating the crew
therapy_crew = Crew(
    agents=[cognitive_analyst, emotional_analyst, topic_segmenter, plan_generator],
    tasks=[segment_task, cognitive_task, emotional_task, cognitive_plan_task, forward_plan_task, summary_task],
    process=Process.sequential  # Execute tasks in the defined order
)




# Kickoff the process with client input
result = therapy_crew.kickoff()
print(result)
