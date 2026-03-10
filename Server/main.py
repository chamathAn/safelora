import eventlet

eventlet.monkey_patch()

from flask import Flask
from flask_socketio import SocketIO, emit
from Agents.orchestrate_agent import OrchestrateAgent
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

import requests

app = Flask(__name__)

socketio = SocketIO(
    app,
    cors_allowed_origins=["http://localhost:3000"],
    async_mode="eventlet",
)


@socketio.on("connect")
def handle_connect():
    print("Client connected")


def run_orchestrator(crop_stage, weather, image):
    checkpointer = SqliteSaver(sqlite3.connect(":memory:", check_same_thread=False))
    orchestrator = OrchestrateAgent(checkpointer=checkpointer)

    initial_state = {
        "image": image,
        "crop_stage": crop_stage,
        "weather": weather,
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
def handle_start(data):
    print("Start received")
    print("Data:", data["cropStage"], data["latitude"], data["longitude"])
    image = data["image"]
    crop_stage = data["cropStage"]
    latitude = data["latitude"]
    longitude = data["longitude"]
    weather = get_weather(latitude, longitude)
    print("Weather:", weather)
    socketio.start_background_task(run_orchestrator, crop_stage, weather, image)


def get_weather(latitude, longitude):
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={latitude}&longitude={longitude}&current_weather=true"
    )

    res = requests.get(url)
    data = res.json()

    cw = data["current_weather"]

    weather_summary = (
        f"Temperature {cw['temperature']}°C, "
        f"Wind speed {cw['windspeed']} km/h, "
        f"Time {cw['time']}"
    )

    return weather_summary


if __name__ == "__main__":
    socketio.run(app, host="127.0.0.1", port=5000)
