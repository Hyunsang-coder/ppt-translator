/**
 * Polling client for job progress updates
 * Replaces SSE to avoid Vercel proxy timeout issues
 */

import type { JobStatusResponse, SSEEvent } from "@/types/api";

export type SSEEventHandler = (event: SSEEvent) => void;

export interface SSEClientOptions {
  onProgress?: SSEEventHandler;
  onComplete?: SSEEventHandler;
  onError?: SSEEventHandler;
  onStarted?: SSEEventHandler;
  onCancelled?: SSEEventHandler;
  /** Polling interval in ms (default: 2000) */
  pollingInterval?: number;
  /** Job status fetcher */
  getJobStatus: (jobId: string) => Promise<JobStatusResponse>;
  /** Job ID */
  jobId: string;
}

export class SSEClient {
  private options: SSEClientOptions;
  private closed = false;
  private pollingTimer: ReturnType<typeof setInterval> | null = null;
  private started = false;

  constructor(_url: string, options: SSEClientOptions) {
    this.options = {
      pollingInterval: 2000,
      ...options,
    };
  }

  connect(): void {
    if (this.closed) return;
    this.startPolling();
  }

  private startPolling(): void {
    if (this.closed) return;

    const poll = async () => {
      if (this.closed) return;

      try {
        const status = await this.options.getJobStatus(this.options.jobId);
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
        break;
      case "running":
        if (!this.started) {
          this.started = true;
          this.options.onStarted?.({
            type: "started",
            data: {},
            timestamp: now,
          });
        }
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

  close(): void {
    this.closed = true;
    if (this.pollingTimer) {
      clearInterval(this.pollingTimer);
      this.pollingTimer = null;
    }
  }
}

export function createSSEClient(url: string, options: SSEClientOptions): SSEClient {
  const client = new SSEClient(url, options);
  client.connect();
  return client;
}
