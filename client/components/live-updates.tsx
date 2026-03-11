"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { io, Socket } from "socket.io-client";

import {
  Reasoning,
  ReasoningContent,
  ReasoningTrigger,
} from "@/components/ai-elements/reasoning";

export default function LiveUpdates() {
  const socketRef = useRef<Socket | null>(null);

  const [content, setContent] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [currentTokenIndex, setCurrentTokenIndex] = useState(0);
  const [tokens, setTokens] = useState<string[]>([]);

  const chunkIntoTokens = useCallback((text: string): string[] => {
    const tokens: string[] = [];
    let i = 0;

    while (i < text.length) {
      const chunkSize = Math.floor(Math.random() * 2) + 3; // 3-4 characters
      tokens.push(text.slice(i, i + chunkSize));
      i += chunkSize;
    }

    return tokens;
  }, []);

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
      const tokenized = chunkIntoTokens(data);

      setTokens(tokenized);
      setContent("");
      setCurrentTokenIndex(0);
      setIsStreaming(true);
    });

    return () => {
      socket.disconnect();
      socketRef.current = null;
    };
  }, [chunkIntoTokens]);

  useEffect(() => {
    if (!isStreaming || currentTokenIndex >= tokens.length) {
      if (isStreaming) {
        setIsStreaming(false);
      }
      return;
    }

    const timer = setTimeout(() => {
      setContent((prev) => prev + tokens[currentTokenIndex]);
      setCurrentTokenIndex((prev) => prev + 1);
    }, 25);

    return () => clearTimeout(timer);
  }, [isStreaming, currentTokenIndex, tokens]);

  return (
    <section>
      <Reasoning className="w-full" isStreaming={isStreaming}>
        <ReasoningTrigger />
        <ReasoningContent>{content}</ReasoningContent>
      </Reasoning>
    </section>
  );
}
