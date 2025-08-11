"""simple workflow"""

from typing import TypedDict

from langgraph.graph import StateGraph, START, END


# state = {}
class BMIState(TypedDict):
    weight_in_kgs: float
    height_in_meters: float
    bmi: float
    category: str


def calculate_bmi(state: BMIState) -> BMIState:
    weight = state["weight_in_kgs"]
    height = state["height_in_meters"]

    bmi = weight / (height**2)
    state["bmi"] = round(bmi, 2)
    return state


def bmi_category(state: BMIState) -> BMIState:
    bmi = state["bmi"]
    if bmi < 18.5:
        state["category"] = "Underweight"
    elif 18.5 <= bmi < 25:
        state["category"] = "Normal"
    elif 25 <= bmi < 30:
        state["category"] = "Overweight"
    else:
        state["category"] = "Obese"

    return state


# 1. define the graph
graph = StateGraph(BMIState)


# 2. add nodes to the graph
graph.add_node("calculate_bmi", calculate_bmi)
graph.add_node("bmi_category", bmi_category)

# 3. add edges to the graph
graph.add_edge(START, "calculate_bmi")
graph.add_edge("calculate_bmi", "bmi_category")
graph.add_edge("bmi_category", END)

# 4. compile the graph
workflow = graph.compile()

# 5. invoke the graph
final_state = workflow.invoke({"weight_in_kgs": 67, "height_in_meters": 1.72})
print(final_state)
