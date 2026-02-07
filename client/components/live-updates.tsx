"use client";

import { useEffect, useRef, useState } from "react";
import { io, Socket } from "socket.io-client";

export default function LiveUpdates() {
  const socketRef = useRef<Socket | null>(null);
  const [messages, setMessages] = useState<string[]>([]);

  useEffect(() => {
    if (socketRef.current) return;

    const socket = io("http://localhost:5000", {
      transports: ["websocket"],
    });

    socketRef.current = socket;

    socket.on("connect", () => {
      console.log("Connected:", socket.id);
    });

    socket.on("update", (data: string) => {
      setMessages((prev) => [...prev, data]);
    });

    return () => {
      socket.disconnect();
      socketRef.current = null;
    };
  }, []);

  return (
    <section>
      <h2>Live Updates</h2>

      <button
        className="px-4 py-2 bg-gray-600"
        onClick={() => socketRef.current?.emit("start")}
      >
        Start
      </button>

      <div>
        {messages.map((msg, i) => (
          <div key={i}>{msg}</div>
        ))}
      </div>
    </section>
  );
}
