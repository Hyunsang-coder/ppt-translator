#!/usr/bin/env node

/**
 * One-time CI/CD bootstrap for Vercel production deploys.
 *
 * Usage:
 *   VERCEL_TOKEN=... node scripts/cicd/bootstrap-vercel.mjs
 *
 * Requires:
 *   - `gh` authenticated for Hyunsang-coder/ppt-translator
 *   - A Vercel token with project read/write scope
 */

const VERCEL_TOKEN = process.env.VERCEL_TOKEN;
const GITHUB_REPO = process.env.GITHUB_REPO ?? "Hyunsang-coder/ppt-translator";
const TEAM_SLUG = process.env.VERCEL_TEAM_SLUG ?? "hyunsang-coders-projects";
const PROJECT_NAME = process.env.VERCEL_PROJECT_NAME ?? "ppt-translator";
const PRODUCTION_BRANCH = "main";
const ROOT_DIRECTORY = "frontend";
const DEPLOY_HOOK_NAME = "github-actions-production";

if (!VERCEL_TOKEN) {
  console.error("Missing VERCEL_TOKEN.");
  console.error("Create one at https://vercel.com/account/tokens and rerun:");
  console.error("  VERCEL_TOKEN=... node scripts/cicd/bootstrap-vercel.mjs");
  process.exit(1);
}

function withTeam(path) {
  const url = new URL(`https://api.vercel.com${path}`);
  url.searchParams.set("slug", TEAM_SLUG);
  return url;
}

async function vercel(path, options = {}) {
  const response = await fetch(withTeam(path), {
    ...options,
    headers: {
      Authorization: `Bearer ${VERCEL_TOKEN}`,
      "Content-Type": "application/json",
      ...(options.headers ?? {}),
    },
  });

  const text = await response.text();
  if (!response.ok) {
    throw new Error(`${response.status} ${path}: ${text}`);
  }

  return text ? JSON.parse(text) : null;
}

async function runGh(args) {
  const { spawnSync } = await import("node:child_process");
  const result = spawnSync("gh", args, {
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
  });

  if (result.status !== 0) {
    throw new Error(result.stderr || result.stdout || `gh ${args.join(" ")} failed`);
  }

  return result.stdout.trim();
}

async function setSecret(name, value) {
  console.log(`Setting GitHub secret ${name}...`);
  await runGh(["secret", "set", name, "--repo", GITHUB_REPO, "--body", value]);
}

async function main() {
  console.log("Verifying Vercel token...");
  const user = await vercel("/v2/user");
  console.log(`Authenticated as ${user.user?.username ?? user.username ?? "unknown"}.`);

  console.log(`Looking up Vercel project ${PROJECT_NAME}...`);
  const projects = await vercel(`/v9/projects?search=${encodeURIComponent(PROJECT_NAME)}`);
  const project =
    projects.projects?.find((entry) => entry.name === PROJECT_NAME) ??
    projects.projects?.[0];

  if (!project?.id) {
    throw new Error(`Could not find Vercel project ${PROJECT_NAME} in team ${TEAM_SLUG}.`);
  }

  const currentRoot = project.rootDirectory ?? null;
  const currentProductionBranch = project.link?.productionBranch ?? null;

  if (currentRoot !== ROOT_DIRECTORY) {
    console.log(`Updating project rootDirectory to ${ROOT_DIRECTORY}...`);
    await vercel(`/v9/projects/${project.id}`, {
      method: "PATCH",
      body: JSON.stringify({
        rootDirectory: ROOT_DIRECTORY,
      }),
    });
  } else {
    console.log(`Project rootDirectory already set to ${ROOT_DIRECTORY}.`);
  }

  if (currentProductionBranch && currentProductionBranch !== PRODUCTION_BRANCH) {
    console.warn(
      `Warning: Vercel git production branch is "${currentProductionBranch}", expected "${PRODUCTION_BRANCH}".`,
    );
    console.warn(
      "Git pushes to main may only create Preview deployments until this is changed in Vercel project settings.",
    );
    console.warn(
      "Deploy hooks and GitHub Actions production deploys will still target main explicitly.",
    );
  }

  console.log("Ensuring production deploy hook exists...");
  const teamId = project.accountId ?? project.teamId;
  const hooksPath = teamId
    ? `/v1/projects/${project.id}/deploy-hooks?teamId=${teamId}`
    : `/v1/projects/${project.id}/deploy-hooks`;

  let hook = project.link?.deployHooks?.find((entry) => entry.name === DEPLOY_HOOK_NAME) ?? null;

  if (!hook) {
    const created = await vercel(hooksPath, {
      method: "POST",
      body: JSON.stringify({
        name: DEPLOY_HOOK_NAME,
        ref: PRODUCTION_BRANCH,
        projectId: project.id,
      }),
    });

    hook =
      created.link?.deployHooks?.find((entry) => entry.name === DEPLOY_HOOK_NAME) ??
      created.deployHooks?.find((entry) => entry.name === DEPLOY_HOOK_NAME) ??
      created;
  }

  if (!hook?.url) {
    const refreshed = await vercel(`/v9/projects/${project.id}`);
    hook =
      refreshed.link?.deployHooks?.find((entry) => entry.name === DEPLOY_HOOK_NAME) ?? null;
  }

  if (!hook?.url) {
    throw new Error("Failed to create or locate a Vercel deploy hook URL.");
  }

  const orgId = project.accountId ?? project.teamId ?? project.orgId;
  if (!orgId) {
    throw new Error("Could not determine VERCEL_ORG_ID from project metadata.");
  }

  await setSecret("VERCEL_TOKEN", VERCEL_TOKEN);
  await setSecret("VERCEL_ORG_ID", orgId);
  await setSecret("VERCEL_PROJECT_ID", project.id);
  await setSecret("VERCEL_DEPLOY_HOOK_URL", hook.url);

  console.log("Triggering initial Vercel production deploy...");
  await fetch(hook.url, { method: "POST" });

  console.log("Triggering Deploy Web workflow...");
  await runGh(["workflow", "run", "deploy-web.yml", "--repo", GITHUB_REPO, "--ref", PRODUCTION_BRANCH]);

  console.log("");
  console.log("CI/CD bootstrap complete.");
  console.log("- GitHub secrets configured: VERCEL_TOKEN, VERCEL_ORG_ID, VERCEL_PROJECT_ID, VERCEL_DEPLOY_HOOK_URL");
  console.log("- Vercel production branch:", PRODUCTION_BRANCH);
  console.log("- Vercel root directory:", ROOT_DIRECTORY);
  console.log("- Deploy Web workflow dispatched");
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exit(1);
});
