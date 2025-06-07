# Cardiovascular Emergency Response System

```
The Cardiovascular Emergency Response system leverages innovative technologies such as Langchain and Langgraph, combined with Redis, to create a robust support framework. Its primary aim is to assist patients promptly by efficiently collecting data from relevant medical documents. By integrating real-time data processing and natural language understanding, the system ensures that critical information is readily accessible to healthcare providers. This approach is designed to enhance response times during stroke emergencies, ultimately contributing to better patient outcomes. Through this solution, we expect to facilitate quicker recovery for individuals experiencing cardiovascular events.
```

---

## Project Overview

The Cardiovascular Emergency Response System is an AI-powered assistant designed to support healthcare professionals and patients during cardiovascular emergencies, such as strokes or heart attacks. The system automates the triage process, asks critical questions, analyzes patient responses, and provides actionable recommendations, including when to call for emergency services.

### Key Features

- **Automated Triage:** The system interacts with patients or caregivers, asking context-aware, medically relevant questions to assess the situation.
- **Real-Time Decision Support:** Utilizes AI models to analyze responses and medical documents, providing instant recommendations.
- **Integration with Medical Protocols:** References up-to-date clinical guidelines and protocols to ensure accurate and safe advice.
- **Data Logging:** Stores patient interactions and responses in Redis for audit, review, and continuous improvement.
- **Extensible Workflow:** Built on Langchain and Langgraph, allowing easy customization and extension for other medical scenarios.

---

## System Architecture

- **Langchain & Langgraph:** Orchestrate the flow of conversation and decision-making logic.
- **Redis:** Stores patient data, interaction history, and system state for fast retrieval and reliability.
- **Google Gemini/OpenAI:** Powers the natural language understanding and content generation.
- **Tkinter UI (optional):** Provides a simple graphical interface for alerts and notifications.

---

## Example Workflow

1. **Start:** The system asks, "Stay alive?" to check for immediate patient responsiveness.
2. **Triage:** If approved, the system proceeds to ask a series of critical yes/no questions.
3. **Analysis:** AI analyzes the answers and determines the severity of the situation.
4. **Action:** The system recommends calling an ambulance or suggests further steps.
5. **Alert:** If the patient is unresponsive or in danger, a visual alert is triggered.

---

## Slides & Presentation

### Slide 1: Introduction
- Problem: Delays in cardiovascular emergency response.
- Solution: AI-powered triage and decision support.

### Slide 2: System Architecture
- Diagram showing Langchain, Langgraph, Redis, and AI model integration.

### Slide 3: Workflow
- Step-by-step flowchart of patient interaction and decision-making.

### Slide 4: Key Features
- Automated triage, real-time support, protocol integration, data logging.

### Slide 5: Impact
- Faster response times, improved patient outcomes, scalable to other emergencies.

### Slide 6: Demo
- Screenshots or live demo of the system in action.

### Slide 7: Future Work
- Integration with hospital systems, multilingual support, advanced analytics.

---

## Getting Started

1. **Install dependencies:**  
   `pip install -r requirements.txt`
2. **Set environment variables:**  
   - `GOOGLE_API_KEY` or `OPENAI_API_KEY`
   - `LANGSMITH_PROJECT=bio`
3. **Run the system:**  
   `python src/agent/bio.py`

---

## Contributing

Contributions are welcome! Please open issues or submit pull requests for new features, bug fixes, or documentation improvements.

---

## License

This project is licensed under the MIT License.

---