import os
import requests
from dotenv import load_dotenv
from langchain.agents import tool
from textwrap import dedent
from crewai import Agent, Crew, Task
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
import streamlit as st

# Load environment variables from .env file
load_dotenv()

# Load API keys
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Initialize Ollama with Llama3
ollama_llm = Ollama(model="llama3.1")

# Define a more structured prompt template
prompt_template = PromptTemplate.from_template("""{backstory}

    You are a helpful AI assistant with the role: {role}.
    Your goal is: {goal}.

    Here is your current task: {task}

    If you need to use a tool, please clearly state the action and the input.

    {agent_scratchpad}""")

class SerperSearchTool:
    @tool
    def search(query: str) -> str:
        """Perform a web search using Serper API.

        Args:
            query (str): The search query

        Returns:
            str: Search results or error message
        """
        headers = {
            "X-API-KEY": SERPER_API_KEY,
            "Content-Type": "application/json"
        }
        body = {"q": query}
        try:
            response = requests.post("https://google.serper.dev/search", headers=headers, json=body)
            data = response.json()
            if "error" in data:
                return f"Serper error: {data['error']}"
            results = data.get("organic", [])
            if not results:
                return "No relevant results found."
            summary = "\n\n".join([f"- {r['title']} - {r['snippet']}" for r in results[:3]])
            return f"Top results:\n\n{summary}"
        except Exception as e:
            return f"Error calling Serper: {e}"

    @classmethod
    def tools(cls):
        return [cls.search]

class OsintAgentsSimplified():
    def CoreInfo_agent(self):
        return Agent(
            role='OSINT Core Information Specialist',
            goal='Gather fundamental information about the company, its online presence, and public perception.',
            tools=SerperSearchTool.tools(),
            backstory=dedent("""\
                     You are a highly efficient OSINT Core Information Specialist. Your primary goal is to quickly gather the most crucial information about a given company. This includes its official website, a brief overview, key social media profiles, recent news, and general public sentiment. You prioritize speed and accuracy in identifying these core elements."""),
            verbose=True,
            llm=ollama_llm,
            prompt=prompt_template,
        )

    def TechnicalAndLegal_agent(self):
        return Agent(
            role='OSINT Technical and Legal Analyst',
            goal='Investigate the company\'s technical infrastructure and identify any significant regulatory or legal information.',
            tools=SerperSearchTool.tools(),
            backstory=dedent("""\
           You are a focused OSINT Technical and Legal Analyst. Your task is to efficiently investigate the technical aspects of a company's online presence, including its domain information and security posture. Additionally, you will look for any readily available information regarding regulatory filings or significant legal issues."""),
            verbose=True,
            llm=ollama_llm,
            prompt=prompt_template,
        )

    def ReportGenerator_agent(self):
        return Agent(
            role='OSINT Report Generator',
            goal='Compile the gathered information into a concise and accurate OSINT report.',
            tools=SerperSearchTool.tools(),
            backstory=dedent("""\
             You are the lead OSINT Report Generator. Your role is to take the key findings from the Core Information Specialist and the Technical and Legal Analyst and synthesize them into a concise yet informative OSINT report. Focus on presenting the most critical information clearly and accurately."""),
            verbose=True,
            llm=ollama_llm,
            prompt=prompt_template,
        )

class OsintAnalysisTaskSimplified():
    def CoreInfo_task(self, agent, company):
        return Task(
            description=dedent(f"""\
                Your primary task is to gather the most important foundational information about the company named {company}. Specifically, find and report:
                - The official website URL.
                - A brief overview or description of what the company does.
                - Links to its main social media profiles (LinkedIn, Twitter, Facebook).
                - A summary of any significant recent news articles (last 3-6 months).
                - A general sense of public perception (e.g., recent customer reviews or mentions on forums).
                Company Name: {company} """),
            expected_output=dedent("""\
                A concise report containing the company's website URL, a brief overview, links to main social media profiles, a summary of recent news, and a general overview of public perception."""),
            async_execution=True,
            agent=agent
        )

    def TechnicalAndLegal_task(self, agent, company):
        return Task(
            description=dedent(f"""\
                Your task is to investigate the technical and legal aspects of the company named {company}. Focus on:
                - Finding the domain registration details (WHOIS information).
                - Briefly assessing the security of their website (e.g., presence of SSL certificate).
                - Identifying any readily available information about significant regulatory filings or major legal issues the company might be facing.
                Company Name: {company}"""),
            expected_output=dedent("""\
                A report summarizing the domain registration details, a brief note on website security, and any identified significant regulatory or legal information."""),
            async_execution=True,
            agent=agent
        )

    def ReportGenerator_task(self, agent, company):
        return Task(
            description=dedent(f"""\
            Your final task is to compile the information gathered by the OSINT Core Information Specialist and the OSINT Technical and Legal Analyst into a concise and accurate OSINT report for the company named {company}. Ensure the report includes:
            - A brief overview of the company.
            - The official website URL.
            - Links to main social media profiles.
            - A summary of recent news.
            - General public perception.
            - Domain registration details.
            - A note on website security.
            - Any identified significant regulatory or legal information.
            Company Name: {company}"""),
            expected_output=dedent("""\
                A concise and accurate OSINT report summarizing the key findings across all investigated areas."""),
            agent=agent
        )

if __name__ == "__main__":
    st.title("Simplified OSINT Analysis Tool")
    st.markdown("Enter the name of the company you want to analyze:")

    company = st.text_input("Company Name", "")

    if company:
        with st.spinner(f"Analyzing {company}..."):
            tasks = OsintAnalysisTaskSimplified()
            agents = OsintAgentsSimplified()
            serper_tool = SerperSearchTool()

            # Instantiate agents with the bound Serper Search Tool
            core_info_agent = agents.CoreInfo_agent()
            core_info_agent.tools = [serper_tool.search]

            technical_legal_agent = agents.TechnicalAndLegal_agent()
            technical_legal_agent.tools = [serper_tool.search]

            report_generator_agent = agents.ReportGenerator_agent()
            report_generator_agent.tools = [serper_tool.search]

            # Create tasks
            core_info_task = tasks.CoreInfo_task(core_info_agent, company)
            technical_legal_task = tasks.TechnicalAndLegal_task(technical_legal_agent, company)
            report_generator_task = tasks.ReportGenerator_task(report_generator_agent, company)

            # Create the crew
            crew = Crew(
                agents=[
                    core_info_agent,
                    technical_legal_agent,
                    report_generator_agent
                ],
                tasks=[
                    core_info_task,
                    technical_legal_task,
                    report_generator_task
                ],
                verbose=True,
                max_iterations=10
            )

            # Run the crew
            report = crew.kickoff()

        st.subheader("OSINT Analysis Report:")
        st.write(report)
        st.success("OSINT analysis complete!")