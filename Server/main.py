from flask import Flask
from Agents.orchestrate_agent import OrchestrateAgent

app = Flask(__name__)


from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3


@app.route("/")
def hello_world():
    checkpointer = SqliteSaver(sqlite3.connect(":memory:", check_same_thread=False))
    orchestrator = OrchestrateAgent(checkpointer=checkpointer)
    initial_state = {
        "plant": "Tomato",
        "disease": "yellow leaf curl virus",
        "crop_stage": "fruit development",
        "weather": "High humidity",
        "messages": [],
    }

    thread = {"configurable": {"thread_id": "1"}}

    for event in orchestrator.graph.stream(initial_state, thread):
        for v in event.values():
            print(v["messages"])

    return "Hello, World! "


if __name__ == "__main__":
    app.run(debug=True)
