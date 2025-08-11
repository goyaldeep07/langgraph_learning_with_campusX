"""
Conditional Workflows without LLM
"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END


class QuadState(TypedDict):
    a: int
    b: int
    c: int

    equation: str
    discriminant: float
    result: str


def show_equation(state: QuadState):
    equation = f'{state["a"]}x2{state["b"]}x{state["c"]}'

    return {"equation": equation}


def calc_discriminant(state: QuadState):
    a = state["a"]
    b = state["b"]
    c = state["c"]

    return {"discriminant": b**2 - (4 * a * c)}


def real_roots(state: QuadState):
    a = state["a"]
    b = state["b"]
    discriminant = state["discriminant"]

    root1 = (-b + (discriminant**0.5)) / 2 * a
    root2 = (-b - (discriminant**0.5)) / 2 * a

    result = f"The roots are {root1} and {root2}"

    return {"result": result}


def repeated_roots(state: QuadState):
    a = state["a"]
    b = state["b"]

    root = (-b) / (2 * a)
    result = f"Only repeating root is {root}"

    return {"result": result}


def no_real_roots(state: QuadState):
    result = f"No real roots"

    return {"result": result}


def check_condition(
    state: QuadState,
) -> Literal["real_roots", "repeated_roots", "no_real_roots"]:
    discriminant = state["discriminant"]

    if discriminant > 0:
        return "real_roots"
    elif discriminant == 0:
        return "repeated_roots"
    else:
        return "no_real_roots"


graph = StateGraph(QuadState)

graph.add_node("show_equation", show_equation)
graph.add_node("calc_discriminant", calc_discriminant)
graph.add_node("real_roots", real_roots)
graph.add_node("no_real_roots", no_real_roots)
graph.add_node("repeated_roots", repeated_roots)


graph.add_edge(START, "show_equation")
graph.add_edge("show_equation", "calc_discriminant")
graph.add_conditional_edges("calc_discriminant", check_condition)
graph.add_edge("real_roots", END)
graph.add_edge("no_real_roots", END)
graph.add_edge("repeated_roots", END)


workflow = graph.compile()

initial_state = {"a": 4, "b": -5, "c": -4}
final_State = workflow.invoke(initial_state)
print(final_State)
