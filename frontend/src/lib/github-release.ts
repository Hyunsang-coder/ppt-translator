const GITHUB_REPO = "Hyunsang-coder/ppt-translator";
const RELEASES_LATEST_URL = `https://api.github.com/repos/${GITHUB_REPO}/releases/latest`;

export type LatestRelease = {
  tagName: string;
  name: string;
  publishedAt: string;
  htmlUrl: string;
};

export function formatReleaseDate(isoDate: string): string {
  return new Intl.DateTimeFormat("ko-KR", {
    year: "numeric",
    month: "long",
    day: "numeric",
    timeZone: "Asia/Seoul",
  }).format(new Date(isoDate));
}

export async function getLatestRelease(): Promise<LatestRelease | null> {
  try {
    const response = await fetch(RELEASES_LATEST_URL, {
      headers: {
        Accept: "application/vnd.github+json",
        "User-Agent": "ppt-translator-web",
      },
      next: { revalidate: 300 },
    });

    if (!response.ok) {
      return null;
    }

    const data = (await response.json()) as {
      tag_name?: string;
      name?: string;
      published_at?: string;
      html_url?: string;
    };

    if (!data.tag_name || !data.published_at || !data.html_url) {
      return null;
    }

    return {
      tagName: data.tag_name,
      name: data.name ?? data.tag_name,
      publishedAt: data.published_at,
      htmlUrl: data.html_url,
    };
  } catch {
    return null;
  }
}
