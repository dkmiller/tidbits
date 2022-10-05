/**
 * Welcome to Cloudflare Workers! This is your first worker.
 *
 * - Run `wrangler dev src/index.ts` in your terminal to start a development server
 * - Open a browser tab at http://localhost:8787/ to see your worker in action
 * - Run `wrangler publish src/index.ts --name my-worker` to publish your worker
 *
 * Learn more at https://developers.cloudflare.com/workers/
 */

import { Octokit } from "@octokit/core";


export interface Env {
	Dan_Kv: KVNamespace;
}

export default {
	async fetch(
		request: Request,
		env: Env,
		ctx: ExecutionContext
	): Promise<Response> {
		const method = request.method;
		// https://stackoverflow.com/a/51865131
		const headers = Array.from(request.headers.entries())
			.map((k, v) => `${k}: ${v}`)
			.join("\n");

		const url = request.url;

		let repoNames = "";
		
		try {
			const pat = await env.Dan_Kv.get("github-pat");

			// https://docs.github.com/en/rest/repos/repos#list-repositories-for-the-authenticated-user
			const octokit = new Octokit({
				auth: pat
			});
	
			const privateRepos = await octokit.request('GET /user/repos', { 'visibility': 'private' });
			repoNames = privateRepos.data.map(t => t.full_name).join(", ");
		} catch (error) {
			// https://stackoverflow.com/a/62611888
			repoNames = error instanceof Error? error.message: `${error}`;
		}

		return new Response(`Hello\n\n${method} ${url}\n\n${headers}\n\n${repoNames}`);
	},
};
