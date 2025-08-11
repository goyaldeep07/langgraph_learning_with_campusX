"""Parallel UPSC Essay Evaluation Workflow using LangGraph and LLM.

This script evaluates a UPSC essay on three independent dimensions:
- Language quality
- Depth of analysis
- Clarity of thought

It runs these evaluations in parallel using LangGraph, collects individual scores,
and then generates a final average score along with a summarized feedback.
"""

import operator
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv


from langgraph.graph import StateGraph, START, END
from langchain_openai.chat_models import ChatOpenAI
from pydantic import BaseModel, Field


load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini")


class EvaluationSchema(BaseModel):
    """
    Pydantic schema for structured LLM output when evaluating essays.

    Attributes:
        feedback (str): Detailed qualitative feedback for the essay.
        score (int): Numerical score between 0 and 10.
    """

    feedback: Annotated[str, Field(description="Detailed feedback for the essay")]
    score: Annotated[int, Field(description="Score out of 10", ge=0, le=10)]


model_with_structured_output = model.with_structured_output(EvaluationSchema)


class UPSCState(TypedDict):
    """
    State structure for the UPSC essay evaluation workflow.

    Attributes:
        essay (str): The essay text to be evaluated.
        language_feedback (str): Feedback on language quality.
        analysis_feedback (str): Feedback on depth of analysis.
        clarity_feedback (str): Feedback on clarity of thought.
        overall_feedback (str): Summarized overall feedback.
        individual_scores (List[int]): Scores from each evaluation dimension.
        avg_score (float): Average score computed from individual scores.
    """

    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_scores: Annotated[List[int], operator.add]
    avg_score: float


def evaluate_language(state: UPSCState):
    """
    Evaluate the language quality of the essay.

    Args:
        state (UPSCState): The current workflow state containing the essay.

    Returns:
        dict: Contains 'language_feedback' and a list with the language score.
    """
    prompt = f"Evaluate the language quality of the following essay and provide a feedback and assign a score out of 10 \n {state['essay']}"
    output = model_with_structured_output.invoke(prompt)
    return {"language_feedback": output.feedback, "individual_scores": [output.score]}


def evaluate_analysis(state: UPSCState):
    """
    Evaluate the depth of analysis of the essay.

    Args:
        state (UPSCState): The current workflow state containing the essay.

    Returns:
        dict: Contains 'analysis_feedback' and a list with the analysis score.
    """
    prompt = f"Evaluate the depth of analysis of the following essay and provide a feedback and assign a score out of 10 \n {state['essay']}"
    output = model_with_structured_output.invoke(prompt)
    return {"analysis_feedback": output.feedback, "individual_scores": [output.score]}


def evaluate_clarity(state: UPSCState):
    """
    Evaluate the clarity of thought in the essay.

    Args:
        state (UPSCState): The current workflow state containing the essay.

    Returns:
        dict: Contains 'clarity_feedback' and a list with the clarity score.
    """
    prompt = f"Evaluate the clarity of though of the following essay and provide a feedback and assign a score out of 10 \n {state['essay']}"
    output = model_with_structured_output.invoke(prompt)
    return {"clarity_feedback": output.feedback, "individual_scores": [output.score]}


def final_evaluation(state: UPSCState):
    """
    Combine feedback from all evaluations and calculate the average score.

    Args:
        state (UPSCState): The current workflow state containing individual feedbacks and scores.

    Returns:
        dict: Contains the overall average score and summarized feedback.
    """
    prompt = (
        f"Based on the following feedback, create a summarized feedback:\n"
        f"Language feedback - {state.get('language_feedback', '')}\n"
        f"Depth of analysis feedback - {state.get('analysis_feedback', '')}\n"
        f"Clarity of thought feedback - {state.get('clarity_feedback', '')}"
    )
    # prompt = f'Based on the following feedbacks create a summarized feedback \n language feedback - {state["language_feedback"]} \n depth of analysis feedback - {state["analysis_feedback"]} \n clarity of thought feedback - {state["clarity_feedback"]}'
    overall_feedback = model.invoke(prompt).content

    avg_score = sum(state["individual_scores"]) / len(state["individual_scores"])
    return {"avg_score": avg_score, "overall_feedback": overall_feedback}


ESSAY = """India in the Age of AI
As the world enters a transformative era defined by artificial intelligence (AI), India stands at a critical juncture — one where it can either emerge as a global leader in AI innovation or risk falling behind in the technology race. The age of AI brings with it immense promise as well as unprecedented challenges, and how India navigates this landscape will shape its socio-economic and geopolitical future.

India's strengths in the AI domain are rooted in its vast pool of skilled engineers, a thriving IT industry, and a growing startup ecosystem. With over 5 million STEM graduates annually and a burgeoning base of AI researchers, India possesses the intellectual capital required to build cutting-edge AI systems. Institutions like IITs, IIITs, and IISc have begun fostering AI research, while private players such as TCS, Infosys, and Wipro are integrating AI into their global services. In 2020, the government launched the National AI Strategy (AI for All) with a focus on inclusive growth, aiming to leverage AI in healthcare, agriculture, education, and smart mobility.

One of the most promising applications of AI in India lies in agriculture, where predictive analytics can guide farmers on optimal sowing times, weather forecasts, and pest control. In healthcare, AI-powered diagnostics can help address India’s doctor-patient ratio crisis, particularly in rural areas. Educational platforms are increasingly using AI to personalize learning paths, while smart governance tools are helping improve public service delivery and fraud detection.

However, the path to AI-led growth is riddled with challenges. Chief among them is the digital divide. While metropolitan cities may embrace AI-driven solutions, rural India continues to struggle with basic internet access and digital literacy. The risk of job displacement due to automation also looms large, especially for low-skilled workers. Without effective skilling and re-skilling programs, AI could exacerbate existing socio-economic inequalities.

Another pressing concern is data privacy and ethics. As AI systems rely heavily on vast datasets, ensuring that personal data is used transparently and responsibly becomes vital. India is still shaping its data protection laws, and in the absence of a strong regulatory framework, AI systems may risk misuse or bias.

To harness AI responsibly, India must adopt a multi-stakeholder approach involving the government, academia, industry, and civil society. Policies should promote open datasets, encourage responsible innovation, and ensure ethical AI practices. There is also a need for international collaboration, particularly with countries leading in AI research, to gain strategic advantage and ensure interoperability in global systems.

India’s demographic dividend, when paired with responsible AI adoption, can unlock massive economic growth, improve governance, and uplift marginalized communities. But this vision will only materialize if AI is seen not merely as a tool for automation, but as an enabler of human-centered development.

In conclusion, India in the age of AI is a story in the making — one of opportunity, responsibility, and transformation. The decisions we make today will not just determine India’s AI trajectory, but also its future as an inclusive, equitable, and innovation-driven society."""


graph = StateGraph(UPSCState)
graph.add_node("evaluate_language", evaluate_language)
graph.add_node("evaluate_analysis", evaluate_analysis)
graph.add_node("evaluate_clarity", evaluate_clarity)
graph.add_node("final_evaluation", final_evaluation)


graph.add_edge(START, "evaluate_language")
graph.add_edge(START, "evaluate_analysis")
graph.add_edge(START, "evaluate_clarity")

graph.add_edge("evaluate_language", "final_evaluation")
graph.add_edge("evaluate_analysis", "final_evaluation")
graph.add_edge("evaluate_clarity", "final_evaluation")

graph.add_edge("final_evaluation", END)

workflow = graph.compile()

initial_State = {"essay": ESSAY}
final_State = workflow.invoke(initial_State)
print(final_State)
