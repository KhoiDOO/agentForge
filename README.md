# AgentForge Kickstart: Multi-Node LangGraph Workflows

This project demonstrates basic LangGraph workflows with multiple nodes and tools integration for the AgentForge Hackathon. 

## Workflows

1. **Research Assistant**: A multi-node workflow that processes research questions, searches for information, analyzes findings, and generates structured reports.
   - Nodes: process_question → search_web → analyze_results → generate_report
   - Features: Web search integration, analysis of search results, structured report generation

## Getting Started

```bash
# Clone the repository
git clone <repository-url>
cd agentForge

# Create isolated environment
python -m venv langgraph-env
source langgraph-env/bin/activate  # On Windows: langgraph-env\Scripts\activate

# Install dependencies
pip install -e . "langgraph-cli[inmem]"

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Run the development server
python -m langgraph_cli dev
```

## 🔑 Get Your API Keys

Before running the project, you need to get free API keys:

1. **OpenAI API Key**
   - Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
   - Create account and generate API key
   - Copy the key (starts with `sk-`)

2. **Tavily API Key** (For web search capabilities)
   - Go to [app.tavily.com](https://app.tavily.com)
   - Sign up and create API key
   - Copy the key (starts with `tvly-`)

3. **LangSmith API Key** (For Tracing)
   - Go to [smith.langchain.com](https://smith.langchain.com)
   - Sign up and get API key
   - Copy the key (starts with `lsv2_`)

## ⚙️ Configure Environment

Edit your `.env` file:

```bash
# Required
OPENAI_API_KEY=sk-your-openai-key-here
TAVILY_API_KEY=tvly-your-tavily-key-here
LANGSMITH_API_KEY=lsv2_your-langsmith-key-here
```

## Running the basic workflow

Once your LangGraph server is up and runnign navigate to localhost link as shown in the image below:
![Alt text](/assets/images/1.png)

Within the input section of the launched graph paste the following JSON:
![Alt text](/assets/images/2.png)

```json
{
  "question": "What is the best rock band in the world",
  "configurable": {
    "temperature": 0.7,
    "max_search_results": 5,
    "search_depth": "basic"
  }
}
```
## Extending the Project

The workflow is designed to be modular and extensible. You can:
- Add new nodes to existing workflows
- Create new tools and integrate them
- Include human in the loop checkpoints where humans can review and approve the results https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/add-human-in-the-loop/
- Include capabilities of memory to store context and tailor preferences https://langchain-ai.github.io/langgraph/concepts/memory/#editing-message-lists

Always hone back into the fact that you want to solve a consumer pain point and automate an exisiting workflow to make it faster, more efficient and reduce the cognitive load on the user.

For TypeScript users and more advanced full stack developers check out this kickstart for a chat UI https://github.com/langchain-ai/agent-chat-ui?tab=readme-ov-file