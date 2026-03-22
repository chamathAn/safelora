"use client";

import { useEffect, useRef, useState } from "react";
import { io, Socket } from "socket.io-client";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export default function Advisory() {
  const socketRef = useRef<Socket | null>(null);
  const [advisory, setAdvisory] = useState("");

  useEffect(() => {
    if (socketRef.current) return;

    const socket = io("http://localhost:5000", {
      transports: ["websocket"],
    });

    socketRef.current = socket;

    socket.on("connect", () => {
      console.log("Connected:", socket.id);
    });

    socket.on("advisory", (data: string) => {
      setAdvisory(data);
    });

    return () => {
      socket.disconnect();
      socketRef.current = null;
    };
  }, []);

  return (
    <div className="size-full">
      {advisory && (
        <h2 className="text-2xl font-bold mb-4">Safelora Advisory</h2>
      )}

      <div className="prose max-w-none">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{advisory}</ReactMarkdown>
      </div>
    </div>
  );
}
