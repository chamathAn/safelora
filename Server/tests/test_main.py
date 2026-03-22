from types import SimpleNamespace
import main


class DummyMessage:
    def __init__(self, content):
        self.content = content


class DummyGraph:
    def __init__(self, events):
        self._events = events

    def stream(self, initial_state, thread):
        return self._events


class DummyOrchestrator:
    def __init__(self, checkpointer=None, events=None):
        self.graph = DummyGraph(events or [])


def test_get_weather_formats_response(monkeypatch):
    class FakeResponse:
        def json(self):
            return {
                "current_weather": {
                    "temperature": 28.5,
                    "windspeed": 12.3,
                    "time": "2026-03-22T10:00",
                }
            }

    def fake_get(url):
        assert "latitude=7.29" in url
        assert "longitude=80.63" in url
        assert "current_weather=true" in url
        return FakeResponse()

    monkeypatch.setattr(main.requests, "get", fake_get)

    result = main.get_weather(7.29, 80.63)

    assert result == "Temperature 28.5°C, Wind speed 12.3 km/h, Time 2026-03-22T10:00"


def test_handle_start_starts_background_task(monkeypatch):
    captured = {}

    def fake_get_weather(lat, lon):
        captured["weather_args"] = (lat, lon)
        return "Temperature 27°C, Wind speed 10 km/h, Time 2026-03-22T11:00"

    def fake_start_background_task(target, crop_stage, weather, image):
        captured["target"] = target
        captured["crop_stage"] = crop_stage
        captured["weather"] = weather
        captured["image"] = image

    monkeypatch.setattr(main, "get_weather", fake_get_weather)
    monkeypatch.setattr(
        main.socketio, "start_background_task", fake_start_background_task
    )

    payload = {
        "cropStage": "fruiting",
        "latitude": 7.1,
        "longitude": 80.1,
        "image": b"fake-image-bytes",
    }

    main.handle_start(payload)

    assert captured["weather_args"] == (7.1, 80.1)
    assert captured["target"] == main.run_orchestrator
    assert captured["crop_stage"] == "fruiting"
    assert (
        captured["weather"]
        == "Temperature 27°C, Wind speed 10 km/h, Time 2026-03-22T11:00"
    )
    assert captured["image"] == b"fake-image-bytes"


def test_run_orchestrator_emits_update(monkeypatch):
    emitted = []

    fake_events = [
        {"context_agent": {"messages": [DummyMessage("Image analysis complete")]}}
    ]

    class FakeOrchestrator:
        def __init__(self, checkpointer=None):
            self.graph = DummyGraph(fake_events)

    monkeypatch.setattr(main, "OrchestrateAgent", FakeOrchestrator)
    monkeypatch.setattr(
        main.socketio,
        "emit",
        lambda event, content: emitted.append((event, content)),
    )

    main.run_orchestrator(
        crop_stage="fruiting",
        weather="Cool and humid",
        image=b"img-bytes",
    )

    assert emitted == [("update", "Image analysis complete")]


def test_run_orchestrator_emits_advisory(monkeypatch):
    emitted = []

    fake_events = [
        {
            "generative_agent": {
                "messages": [DummyMessage("Case Summary\nTomato disease advisory")]
            }
        }
    ]

    class FakeOrchestrator:
        def __init__(self, checkpointer=None):
            self.graph = DummyGraph(fake_events)

    monkeypatch.setattr(main, "OrchestrateAgent", FakeOrchestrator)
    monkeypatch.setattr(
        main.socketio,
        "emit",
        lambda event, content: emitted.append((event, content)),
    )

    main.run_orchestrator(
        crop_stage="vegetative",
        weather="Warm and wet",
        image=b"img-bytes",
    )

    assert emitted == [("advisory", "Case Summary\nTomato disease advisory")]


def test_socket_connect_event_with_test_client():
    client = main.socketio.test_client(main.app)
    assert client.is_connected()
    client.disconnect()
