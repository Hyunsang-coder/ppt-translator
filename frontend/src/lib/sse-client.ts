/**
 * SSE Client for real-time job progress updates
 * Falls back to polling when SSE connection fails (e.g. Vercel proxy timeout)
 */

import type { JobStatusResponse, SSEEvent } from "@/types/api";

export type SSEEventHandler = (event: SSEEvent) => void;

export interface SSEClientOptions {
  onProgress?: SSEEventHandler;
  onComplete?: SSEEventHandler;
  onError?: SSEEventHandler;
  onStarted?: SSEEventHandler;
  onCancelled?: SSEEventHandler;
  reconnectAttempts?: number;
  reconnectDelay?: number;
  /** Polling interval in ms when falling back from SSE (default: 2000) */
  pollingInterval?: number;
  /** Job status fetcher for polling fallback */
  getJobStatus?: (jobId: string) => Promise<JobStatusResponse>;
  /** Job ID for polling fallback */
  jobId?: string;
}

export class SSEClient {
  private eventSource: EventSource | null = null;
  private url: string;
  private options: SSEClientOptions;
  private reconnectCount = 0;
  private closed = false;
  private pollingTimer: ReturnType<typeof setInterval> | null = null;
  private isPolling = false;

  constructor(url: string, options: SSEClientOptions = {}) {
    this.url = url;
    this.options = {
      reconnectAttempts: 3,
      reconnectDelay: 1000,
      pollingInterval: 2000,
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
        // SSE failed — try polling fallback
        if (this.options.getJobStatus && this.options.jobId) {
          console.warn("SSE connection failed, falling back to polling");
          this.startPolling();
        } else {
          this.options.onError?.({
            type: "error",
            data: { message: "Connection lost after multiple reconnect attempts" },
            timestamp: Date.now() / 1000,
          });
        }
      }
    };

    this.eventSource.onopen = () => {
      this.reconnectCount = 0;
    };
  }

  private startPolling(): void {
    if (this.closed || this.isPolling) return;
    this.isPolling = true;

    const poll = async () => {
      if (this.closed) return;

      try {
        const status = await this.options.getJobStatus!(this.options.jobId!);
        this.handlePolledStatus(status);
      } catch (err) {
        console.error("Polling failed:", err);
        // Continue polling — transient network errors shouldn't stop us
      }
    };

    // Poll immediately, then on interval
    poll();
    this.pollingTimer = setInterval(poll, this.options.pollingInterval);
  }

  private handlePolledStatus(status: JobStatusResponse): void {
    if (this.closed) return;

    const now = Date.now() / 1000;

    switch (status.state) {
      case "pending":
        // Job not started yet
        break;
      case "running":
        if (status.progress) {
          this.options.onProgress?.({
            type: "progress",
            data: status.progress as unknown as Record<string, unknown>,
            timestamp: now,
          });
        }
        break;
      case "completed":
        this.options.onComplete?.({
          type: "complete",
          data: {},
          timestamp: now,
        });
        this.close();
        break;
      case "failed":
        this.options.onError?.({
          type: "error",
          data: { message: status.error_message || "번역 중 오류가 발생했습니다." },
          timestamp: now,
        });
        this.close();
        break;
      case "cancelled":
        this.options.onCancelled?.({
          type: "cancelled",
          data: {},
          timestamp: now,
        });
        this.close();
        break;
    }
  }

  private handleEvent(event: SSEEvent): void {
    // Prevent stale events from firing after close() (e.g. retranslate race condition)
    if (this.closed) return;

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
    if (this.pollingTimer) {
      clearInterval(this.pollingTimer);
      this.pollingTimer = null;
    }
    this.isPolling = false;
  }
}

export function createSSEClient(url: string, options: SSEClientOptions): SSEClient {
  const client = new SSEClient(url, options);
  client.connect();
  return client;
}
