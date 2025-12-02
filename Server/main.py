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
        "disease": "Late Blight",
        "crop_stage": "fruiting",
        "weather": "Cool nights with heavy moisture",
        "messages": [],
    }

    thread = {"configurable": {"thread_id": "1"}}
    arr = []
    for event in orchestrator.graph.stream(initial_state, thread):
        for v in event.values():
            print(v["messages"])
            for message in v["messages"]:
                if hasattr(message, "content"):
                    arr.append(message)

    print("=" * 80 + "\n Ended....\n" + "=" * 80)
    print(arr[-2].content) if len(arr) > 0 else ""
    return "Hello, World! "


if __name__ == "__main__":
    app.run(debug=True)
