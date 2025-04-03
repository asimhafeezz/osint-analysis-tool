import os
from exa_py import Exa
from dotenv import load_dotenv
from langchain.agents import tool
from textwrap import dedent
from crewai import Agent, Crew, Task
from langchain_community.llms import Ollama  # NEW IMPORT
from langchain_core.prompts import PromptTemplate # NEW IMPORT

# Load environment variables from .env file
load_dotenv()

# Load API keys from environment variables or .env file
os.environ["EXA_API_KEY"] = os.getenv("EXA_API_KEY")

# Initialize Ollama with Llama3 (ensure you have it pulled: `ollama pull llama3`)
ollama_llm = Ollama(model="llama3.1") # NEW CODE

prompt_template = PromptTemplate.from_template("""{backstory}

    You are a helpful AI assistant collaborating with other agents to achieve a common goal.

    Role: {role}
    Goal: {goal}

    Current task: {task}

    {agent_scratchpad}""") # NEW CODE

class ExaSearchTool:
    @tool
    def search(query: str):
        """Search for a webpage based on the query."""
        results = ExaSearchTool._exa().search(f"{query}", use_autoprompt=True, num_results=10).results
        # Adapt the results to the expected format
        formatted_results = []
        for result in results:
            formatted_results.append({
                "title": result.title,
                "url": result.url,
                "id": result.id
            })
        return formatted_results

    @tool
    def find_similar(url: str):
        """Search for webpages similar to a given URL.
        The url passed in should be a URL returned from `search`.
        """
        results = ExaSearchTool._exa().find_similar(url, num_results=10)
        formatted_results = []
        for result in results:
            formatted_results.append({
                "title": result.title,
                "url": result.url,
                "id": result.id
            })
        return formatted_results

    @tool
    def get_contents(ids: str):
        """Get the contents of a webpage.
        The ids must be passed in as a list, a list of ids returned from `search`.
        """
        try:
            ids = eval(ids)
            contents_response = ExaSearchTool._exa().get_contents(ids)
            contents = []
            if contents_response.results:
                for result in contents_response.results:
                    contents.append(f"Title: {result.title}\nURL: {result.url}\nContent: {result.text[:1000] if result.text else 'No content'}")
            return "\n\n".join(contents)
        except Exception as e:
            return f"Error processing content IDs: {e}"

    @classmethod
    def tools(cls):
        return [cls.search, cls.find_similar, cls.get_contents]

    @staticmethod
    def _exa():
        return Exa(api_key=os.environ["EXA_API_KEY"])

class OsintAgents():
    def CompanyInfo_agent(self):
        return Agent(
            role='Company Information Specialist',
            goal='Conduct thorough research on the company,its website, founded date, founders, Headquarter location, industry and Subsidiaries.',
            tools=ExaSearchTool.tools(),
            backstory=dedent("""\
                     As a Company Research Specialist, your mission is to uncover detailed information
            about the company, what is the company is all about, its official website and url of website, founded date of the company, founders names, location of the headquarter, industry of the company and its subsidiaries.
            Your insights will provide a comprehensive overview of the company's background."""),
            verbose=True,
            llm=ollama_llm,      # UPDATED/ADDED LINE
            prompt=prompt_template, # UPDATED/ADDED LINE,
            max_iterations=5
        )

    def WebsiteAnalysis_agent(self):
        return Agent(
            role='Website Analyst',
            goal='Analyze the official Website of the company its key information',
            tools=ExaSearchTool.tools(),
            backstory=dedent("""\
                     As a Website Analyst, your analysis will focus on the company's website Structure like its sections and pages information, its proper metadata analysis, use tools like BuiltWith to identify Technology stack used in the company's website. Also find out the SSL/TLS configuration.
           Your will provide the detailed analysis of the companys website."""),
            verbose=True,
            llm=ollama_llm,      # UPDATED/ADDED LINE
            prompt=prompt_template, # UPDATED/ADDED LINE,
            max_iterations=5
        )

    def NetworkAnalysis_agent(self):
        return Agent(
            role='Domain and Network Analyst',
            goal='Analyze the Domain Registration Details, DNS Records, Subdomains, IP Address official Website and Network Services.',
            tools=ExaSearchTool.tools(),
            backstory=dedent("""\
           As Domain and Network Analyst, your analyis will focus on Domain Registration Details of the website use tools like whois, you will provide the DNS records using DNSdumster, you will provide the subdomains by using google dorking techniques,
           you will also provide the IP Address associated with the domain and provide the Network services by using the tool shodan."""),
            verbose=True,
            llm=ollama_llm,      # UPDATED/ADDED LINE
            prompt=prompt_template, # UPDATED/ADDED LINE,
            max_iterations=5
        )

    def SocialMediaAndContact_agent(self):
        return Agent(
            role='Social Media and Contact Information Specialist',
            goal='Gather detailed information about the company\'s social media presence and contact details, including phone numbers, email addresses, and key personnel contact information.',
            tools=ExaSearchTool.tools(),
            backstory=dedent("""\
           As a Social Media and Contact Information Specialist, your mission is to uncover the company\'s presence on social media platforms like LinkedIn, Facebook, Twitter, GitHub, Instagram, and others.
                 Additionally, gather comprehensive contact information including phone numbers, email addresses, and any available contact details of key personnel working in the organization."""),
            verbose=True,
            llm=ollama_llm,      # UPDATED/ADDED LINE
            prompt=prompt_template, # UPDATED/ADDED LINE,
            max_iterations=5
        )

    def SearchEngineIntelligence_agent(self):
        return Agent(
            role='Search Engine Intelligence Specialist',
            goal='Use Google Dorking techniques to uncover hidden information like PDFs and confidential files, and find recent news articles about the company.',
            tools=ExaSearchTool.tools(),
            backstory=dedent("""\
           As a Search Engine Intelligence Specialist, your mission is to uncover hidden information about the company using Google Dorking techniques.
             You will also gather and analyze recent news articles about the company, deriving conclusions from the findings."""),
            verbose=True,
            llm=ollama_llm,      # UPDATED/ADDED LINE
            prompt=prompt_template, # UPDATED/ADDED LINE,
            max_iterations=5
        )

    def BusinessInformation_agent(self):
        return Agent(
            role='Business Information Specialist',
            goal='Gather comprehensive business information about the company including company overview, financial information, key personnel, and partnerships',
            tools=ExaSearchTool.tools(),
            backstory=dedent("""\
           As a Business Information Specialist, your mission is to gather detailed business information from various sources.
             You will provide an overview of the company, financial data, information about key personnel, and details of company partnerships."""),
            verbose=True,
            llm=ollama_llm,      # UPDATED/ADDED LINE
            prompt=prompt_template, # UPDATED/ADDED LINE,
            max_iterations=5
        )

    def RegulatoryLegalTechnicalFootprint_agent(self):
        return Agent(
            role='Regulatory, Legal, and Technical Footprint Specialist',
            goal='Gather information about the company’s regulatory filings, legal issues, security posture, and email patterns.',
            tools=ExaSearchTool.tools(),
            backstory=dedent("""\
           As a Regulatory, Legal, and Technical Footprint Specialist, your mission is to gather information on the company’s regulatory filings and legal issues, and assess its technical footprint.
             This includes identifying vulnerabilities and email patterns using various tools."""),
            verbose=True,
            llm=ollama_llm,      # UPDATED/ADDED LINE
            prompt=prompt_template, # UPDATED/ADDED LINE,
            max_iterations=5
        )

    def IntellectualProperty_agent(self):
        return Agent(
            role='Intellectual Property Specialist',
            goal='Gather information about the company’s patents, trademarks, and copyrights.',
            tools=ExaSearchTool.tools(),
            backstory=dedent("""\
           As an Intellectual Property Specialist, your mission is to uncover and document the company’s intellectual property assets.
             This includes registered patents, trademarks, and significant copyrights."""),
            verbose=True,
            llm=ollama_llm,      # UPDATED/ADDED LINE
            prompt=prompt_template, # UPDATED/ADDED LINE,
            max_iterations=5
        )

    def EmployeeHiringInformation_agent(self):
        return Agent(
            role='Employee and Hiring Information Specialist',
            goal='Gather information about current job listings and employee reviews of the company.',
            tools=ExaSearchTool.tools(),
            backstory=dedent("""\
           As an Employee and Hiring Information Specialist, your mission is to gather details about the company’s hiring practices and employee experiences.
             This includes current job openings and reviews from sites like Glassdoor."""),
            verbose=True,
            llm=ollama_llm,      # UPDATED/ADDED LINE
            prompt=prompt_template, # UPDATED/ADDED LINE,
            max_iterations=5
        )

    def CommunityPublicPerception_agent(self):
        return Agent(
            role='Community and Public Perception Specialist',
            goal='Gather customer reviews and forum discussions related to the company.',
            tools=ExaSearchTool.tools(),
            backstory=dedent("""\
             As a Community and Public Perception Specialist, your mission is to gather and analyze public opinions about the company.
             This includes customer reviews from various platforms and forum discussions."""),
            verbose=True,
            llm=ollama_llm,      # UPDATED/ADDED LINE
            prompt=prompt_template, # UPDATED/ADDED LINE,
            max_iterations=5
        )

    def OSINTReportGenerator_agent(self):
        return Agent(
            role='OSINT Report Generator',
            goal='Compile all gathered information into a detailed and comprehensive OSINT report.',
            tools=ExaSearchTool.tools(),
            backstory=dedent("""\
             As the OSINT Report Generator, your role is to consolidate the information from various agents, including Company Information, Website Analysis, Domain and Network Analysis, Social Media and Contact Information, Search Engine Intelligence, Business Information, Regulatory and Legal Information, Technical Footprint, Intellectual Property, Employee and Hiring Information, Community and Public Perception, into a detailed and comprehensive OSINT report.
             This report will provide a thorough overview and analysis of the company."""),
            verbose=True,
                        llm=ollama_llm,      # UPDATED/ADDED LINE
            prompt=prompt_template, # UPDATED/ADDED LINE,
            max_iterations=5
        )

class OsintAnalysisTask():
    def CompanyInfo_task(self, agent, company):
        return Task(
            description=dedent(f"""\
                Conduct comprehensive research on the company named {company}. Gather information about its website, founded date of company, its founders, headquarter location of the company, industry of the company and its subsidiaries.
                Company Name : {company} """),
            expected_output=dedent("""\
                A detailed report summarizing key findings about the company,its website, founded date, founders, headquarter,industry and its subsidiaries."""),
            async_execution=True,
            agent=agent
        )

    def WebsiteAnalysis_task(self, agent, company):
        return Task(
            description=dedent(f"""\
                Uncover information about the website of company named {company}. find its official website and its structure, review the content of the website, make analysis on its metadata, find the technology stack used in the website take the help fo tools like buitWith to find technology stack, also find the SSL/TLS configuration.

                Company Name: {company}"""),
            expected_output=dedent("""\
                A comprehensive report on the company's Website Structure,content review, metadata analysis, technology stack and SSL/TLS configuration."""),
            async_execution=True,
            agent=agent
        )

    def NetworkAnalysis_task(self, agent, company):
        return Task(
            description=dedent(f"""\
                Uncover information about the Domain name and Network of company named {company}.
        find its domain registration details using tool like 'whois',
        find its dns records using tools like 'DNSdumpster',
        find tis subdomains using dns tools and google dorking techniques,
        find its ip addrress associated with domain and
        find different network services using shodan.

        Company Name: {company}"""),
            expected_output=dedent("""\
                A comprehensive report on the company's domain registration details, dns records, subdomains, IP addresses and network services."""),
            async_execution=True,
            agent=agent
        )

    def SocialMediaAndContact_task(self, agent, company):
        return Task(
            description=dedent(f"""\
                Gather detailed information about the social media presence and contact details of the company named {company}.
                Find links to its profiles on platforms like LinkedIn, Facebook, Twitter, GitHub, Instagram, and others.
                Additionally, gather comprehensive contact information including phone numbers, email addresses, and contact details of key personnel if available.
                Company Name: {company}"""),
            expected_output=dedent("""\
                A comprehensive report on the company's social media profiles and contact details, including phone numbers, email addresses, and contact information of key personnel."""),
            async_execution=True,
            agent=agent
        )

    def SearchEngineIntelligence_task(self, agent, company):
        return Task(
            description=dedent(f"""\
            Use Google Dorking techniques to uncover hidden information like PDFs and confidential files about the company named {company}.
            Also, find recent news articles about the company, analyze them, and derive conclusions.
            Company Name: {company}"""),
            expected_output=dedent("""\
            A report on hidden information uncovered using Google Dorking and a summary of recent news articles with analysis and conclusions."""),
            async_execution=True,
            agent=agent
        )

    def BusinessInformation_task(self, agent, company):
        return Task(
            description=dedent(f"""\
            Gather comprehensive business information about the company named {company}.
            This includes an overview from business directories like Bloomberg, Crunchbase, LinkedIn, financial information, key personnel details, and information about company partnerships.
            Company Name: {company}"""),
            expected_output=dedent("""\
            A detailed report summarizing the company's business information, including company overview, financial data, key personnel, and partnerships"""),
            async_execution=True,
            agent=agent
        )

    def RegulatoryLegalTechnicalFootprint_task(self, agent, company):
        return Task(
            description=dedent(f"""\
            Gather information about the regulatory filings, legal issues, security posture, and email patterns of the company named {company}.
            This includes checking filings with bodies like the SEC, identifying ongoing or past legal issues, assessing vulnerabilities using tools like Shodan, and finding company email patterns using tools like Hunter.io.
            Company Name: {company}"""),
            expected_output=dedent("""\
                A comprehensive report on the company's regulatory filings, legal issues, security posture, and email patterns."""),
            async_execution=True,
            agent=agent
        )

    def IntellectualProperty_task(self, agent, company):
        return Task(
            description=dedent(f"""\
            Gather information about the intellectual property of the company named {company}.
            This includes any patents registered by the company, trademarks, and significant copyrights.
            Company Name: {company}"""),
            expected_output=dedent("""\
                A detailed report on the company's intellectual property, including patents, trademarks, and copyrights."""),
            async_execution=True,
            agent=agent
        )

    def EmployeeHiringInformation_task(self, agent, company):
        return Task(
            description=dedent(f"""\
            Gather information about current job listings and employee reviews for the company named {company}.
            This includes job openings from the company’s career page and other job boards, as well as employee reviews from sites like Glassdoor.
            Company Name: {company}"""),
            expected_output=dedent("""\
                A comprehensive report on the company's current job listings and employee reviews"""),
            async_execution=True,
            agent=agent
        )

    def CommunityPublicPerception_task(self, agent, company):
        return Task(
            description=dedent(f"""\
            Gather customer reviews and forum discussions related to the company named {company}.
            This includes reviews from websites like Trustpilot and Google Reviews, and mentions on forums like Reddit and industry-specific forums.
            Company Name: {company}"""),
            expected_output=dedent("""\
                A report on the company's community and public perception, including customer reviews and forum discussions."""),
            async_execution=True,
            agent=agent
        )

    def OSINTReportGenerator_task(self, agent, company):
        return Task(
            description=dedent(f"""\
            Compile all the gathered information, including Company Information, Website Analysis, Domain and Network Analysis, Social Media and Contact Information, Search Engine Intelligence, Business Information, Regulatory and Legal Information, Technical Footprint, Intellectual Property, Employee and Hiring Information, Community and Public Perception, into a concise and comprehensive OSINT report for the company.
            Ensure the report is detailed, well-structured, and provides valuable insights.
            Company Name: {company}"""),
            expected_output=dedent("""\
                A detailed and well-structured OSINT report for the company, including sections on Company Information, Website Analysis, Domain and Network Analysis, Social Media and Contact Information, Search Engine Intelligence, Business Information, Regulatory and Legal Information, Technical Footprint, Intellectual Property, Employee and Hiring Information, Community and Public Perception."""),
            agent=agent
        )

if __name__ == "__main__":
    tasks = OsintAnalysisTask()
    agents = OsintAgents()

    print("## Welcome to OSINT Analysis of Company")
    print('-------------------------------')
    company = input("Enter the name of the company: ")

    # Instantiate agents
    company_info_agent = agents.CompanyInfo_agent()
    website_analysis_agent = agents.WebsiteAnalysis_agent()
    network_analysis_agent = agents.NetworkAnalysis_agent()
    social_media_contact_agent = agents.SocialMediaAndContact_agent()
    search_engine_intel_agent = agents.SearchEngineIntelligence_agent()
    business_info_agent = agents.BusinessInformation_agent()
    regulatory_legal_tech_agent = agents.RegulatoryLegalTechnicalFootprint_agent()
    intellectual_property_agent = agents.IntellectualProperty_agent()
    employee_hiring_agent = agents.EmployeeHiringInformation_agent()
    community_perception_agent = agents.CommunityPublicPerception_agent()
    report_generator_agent = agents.OSINTReportGenerator_agent()

    # Create tasks
    company_info_task = tasks.CompanyInfo_task(company_info_agent, company)
    website_analysis_task = tasks.WebsiteAnalysis_task(website_analysis_agent, company)
    network_analysis_task = tasks.NetworkAnalysis_task(network_analysis_agent, company)
    social_media_contact_task = tasks.SocialMediaAndContact_task(social_media_contact_agent, company)
    search_engine_intel_task = tasks.SearchEngineIntelligence_task(search_engine_intel_agent, company)
    business_info_task = tasks.BusinessInformation_task(business_info_agent, company)
    regulatory_legal_tech_task = tasks.RegulatoryLegalTechnicalFootprint_task(regulatory_legal_tech_agent, company)
    intellectual_property_task = tasks.IntellectualProperty_task(intellectual_property_agent, company)
    employee_hiring_task = tasks.EmployeeHiringInformation_task(employee_hiring_agent, company)
    community_perception_task = tasks.CommunityPublicPerception_task(community_perception_agent, company)
    report_generator_task = tasks.OSINTReportGenerator_task(report_generator_agent, company)

    # Create the crew
    crew = Crew(
        agents=[
            company_info_agent,
            website_analysis_agent,
            network_analysis_agent,
            social_media_contact_agent,
            search_engine_intel_agent,
            business_info_agent,
            regulatory_legal_tech_agent,
            intellectual_property_agent,
            employee_hiring_agent,
            community_perception_agent,
            report_generator_agent
        ],
        tasks=[
            company_info_task,
            website_analysis_task,
            network_analysis_task,
            social_media_contact_task,
            search_engine_intel_task,
            business_info_task,
            regulatory_legal_tech_task,
            intellectual_property_task,
            employee_hiring_task,
            community_perception_task,
            report_generator_task
        ],
        verbose=True,  # You can set it to False if you don't want to see detailed execution logs
        max_iterations=10  # Maximum number of iterations for the crew to run
    )

    # Run the crew
    report = crew.kickoff()

    print("\n\n-------------------------------")
    print("## OSINT Analysis Report:")
    print("-------------------------------")
    print(report)
    print("\n\nOSINT analysis complete.")