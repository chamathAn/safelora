import eventlet

eventlet.monkey_patch()

from flask import Flask
from flask_socketio import SocketIO, emit
from Agents.orchestrate_agent import OrchestrateAgent
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

app = Flask(__name__)

socketio = SocketIO(
    app,
    cors_allowed_origins=["http://localhost:3000"],
    async_mode="eventlet",
)


@socketio.on("connect")
def handle_connect():
    print("Client connected")


def run_orchestrator():
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

    for event in orchestrator.graph.stream(initial_state, thread):
        for v in event.values():
            print(v["messages"])
            for message in v["messages"]:
                if hasattr(message, "content"):
                    # print("EMIT:", message.content)
                    socketio.emit("update", message.content)


@socketio.on("start")
def handle_start():
    print("Start received")
    socketio.start_background_task(run_orchestrator)


if __name__ == "__main__":
    socketio.run(app, host="127.0.0.1", port=5000)
