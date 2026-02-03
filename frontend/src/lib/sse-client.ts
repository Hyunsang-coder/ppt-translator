/**
 * SSE Client for real-time job progress updates
 */

import type { SSEEvent } from "@/types/api";

export type SSEEventHandler = (event: SSEEvent) => void;

export interface SSEClientOptions {
  onProgress?: SSEEventHandler;
  onComplete?: SSEEventHandler;
  onError?: SSEEventHandler;
  onStarted?: SSEEventHandler;
  onCancelled?: SSEEventHandler;
  reconnectAttempts?: number;
  reconnectDelay?: number;
}

export class SSEClient {
  private eventSource: EventSource | null = null;
  private url: string;
  private options: SSEClientOptions;
  private reconnectCount = 0;
  private closed = false;

  constructor(url: string, options: SSEClientOptions = {}) {
    this.url = url;
    this.options = {
      reconnectAttempts: 3,
      reconnectDelay: 1000,
      ...options,
    };
  }

  connect(): void {
    if (this.closed) return;

    this.eventSource = new EventSource(this.url);

    this.eventSource.onmessage = (event) => {
      try {
        const parsed: SSEEvent = JSON.parse(event.data);
        this.handleEvent(parsed);
      } catch (err) {
        console.error("Failed to parse SSE event:", err);
      }
    };

    this.eventSource.onerror = () => {
      if (this.closed) return;

      this.eventSource?.close();

      if (this.reconnectCount < (this.options.reconnectAttempts ?? 3)) {
        this.reconnectCount++;
        setTimeout(() => this.connect(), this.options.reconnectDelay);
      } else {
        this.options.onError?.({
          type: "error",
          data: { message: "Connection lost after multiple reconnect attempts" },
          timestamp: Date.now() / 1000,
        });
      }
    };

    this.eventSource.onopen = () => {
      this.reconnectCount = 0;
    };
  }

  private handleEvent(event: SSEEvent): void {
    switch (event.type) {
      case "progress":
        this.options.onProgress?.(event);
        break;
      case "complete":
        this.options.onComplete?.(event);
        this.close();
        break;
      case "error":
        this.options.onError?.(event);
        this.close();
        break;
      case "started":
        this.options.onStarted?.(event);
        break;
      case "cancelled":
        this.options.onCancelled?.(event);
        this.close();
        break;
      case "keepalive":
        // Ignore keepalive events
        break;
    }
  }

  close(): void {
    this.closed = true;
    this.eventSource?.close();
    this.eventSource = null;
  }
}

export function createSSEClient(url: string, options: SSEClientOptions): SSEClient {
  const client = new SSEClient(url, options);
  client.connect();
  return client;
}
