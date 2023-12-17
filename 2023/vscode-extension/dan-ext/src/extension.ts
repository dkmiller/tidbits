// The module 'vscode' contains the VS Code extensibility API
// Import the module and reference it with the alias vscode in your code below
import * as vscode from 'vscode';
import http from "http";
import url from "url";


// TODO: better way of handling and disposing of state like this.
var servers: Array<http.Server> = [];


// This method is called when your extension is activated
// Your extension is activated the very first time the command is executed
export function activate(context: vscode.ExtensionContext) {

	// Use the console to output diagnostic information (console.log) and errors (console.error)
	// This line of code will only be executed once when your extension is activated
	console.log('Hi from the extension "dan-ext"!');

	// The command has been defined in the package.json file
	// Now provide the implementation of the command with registerCommand
	// The commandId parameter must match the command field in package.json
	let disposable = vscode.commands.registerCommand('dan-ext.danLogin', () => {
		// The code you place here will be executed every time your command is executed

		var port = 1234;
		var parameter = 'code';
		var originalParameterValue = 'Dan_Extension_Parameter_Value';

		// Imitate the MSAL LoopbackClient to create a server for handling
		// the redirect.
		// https://github.com/AzureAD/microsoft-authentication-library-for-js/blob/648501e4d9c65cb39322f97a9e09c5f93e1135b7/lib/msal-node/src/network/LoopbackClient.ts#L36
		var server = http.createServer(
			(req: http.IncomingMessage, res: http.ServerResponse) => {
				console.log('req.url', req.url);
				// Parse params: https://stackoverflow.com/a/47416843
				var params = url.parse(req.url || '', true).query;
				var parameterValue = params[parameter];
				console.log('Parameter value is: ', parameterValue);

				// TODO: find a way of passing the obtained parameter back to
				// the client.
				res.end(`You have logged into Dan!\n\nYour ${parameter} = '${parameterValue}'`);
				return;
			}
		);
		server.listen(port);
		servers.push(server);

		// Httpbun redirects: https://httpbun.com/#redirects
		// Uri.parse is difficult to work with when there are nested query
		// parameters. Instead, we construct the URI to open directly.
		var uri = vscode.Uri.from({
			scheme: 'http',
			authority: `localhost:${port}`,
			path: '/',
			query: `code=${originalParameterValue}`
		});
		console.log('uri', uri.toString());
		// Open a login: https://stackoverflow.com/a/44967017
		vscode.env.openExternal(uri);

		// Display a message box to the user
		vscode.window.showInformationMessage('Login from dan-ext!');
	});

	context.subscriptions.push(disposable);
}

// This method is called when your extension is deactivated
export function deactivate() {
	for (var server of servers) {
		console.log('Closing server', server);
		server.close();
	}
}
