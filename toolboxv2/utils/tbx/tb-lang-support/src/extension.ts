// file: tb-lang-vscode/src/extension.ts

import * as vscode from 'vscode';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

export function activate(context: vscode.ExtensionContext) {
    console.log('TB Language extension activated');

    // Register commands
    context.subscriptions.push(
        vscode.commands.registerCommand('tb.run', runTBProgram),
        vscode.commands.registerCommand('tb.runStreaming', runTBProgramStreaming),
        vscode.commands.registerCommand('tb.compile', compileTBProgram),
        vscode.commands.registerCommand('tb.check', checkTBProgram)
    );

    // Register hover provider
    context.subscriptions.push(
        vscode.languages.registerHoverProvider('tb', new TBHoverProvider())
    );

    // Register completion provider
    context.subscriptions.push(
        vscode.languages.registerCompletionItemProvider('tb', new TBCompletionProvider(), '.', '$')
    );
}

async function runTBProgram() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showErrorMessage('No active editor');
        return;
    }

    const document = editor.document;
    if (document.languageId !== 'tb') {
        vscode.window.showErrorMessage('Not a TB file');
        return;
    }

    await document.save();

    const config = vscode.workspace.getConfiguration('tb');
    const executable = config.get<string>('executablePath') || 'tb';
    const mode = config.get<string>('defaultMode') || 'jit';

    const terminal = vscode.window.createTerminal('TB Program');
    terminal.show();
    terminal.sendText(`${executable} run "${document.uri.fsPath}" --mode ${mode}`);
}

async function runTBProgramStreaming() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) return;

    await editor.document.save();

    const config = vscode.workspace.getConfiguration('tb');
    const executable = config.get<string>('executablePath') || 'tb';

    const terminal = vscode.window.createTerminal('TB Program (Streaming)');
    terminal.show();
    terminal.sendText(`${executable} run "${editor.document.uri.fsPath}" --mode streaming`);
}

async function compileTBProgram() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) return;

    await editor.document.save();

    const outputPath = await vscode.window.showInputBox({
        prompt: 'Output file path',
        value: editor.document.uri.fsPath.replace('.tbx', '')
    });

    if (!outputPath) return;

    const config = vscode.workspace.getConfiguration('tb');
    const executable = config.get<string>('executablePath') || 'tb';

    const terminal = vscode.window.createTerminal('TB Compile');
    terminal.show();
    terminal.sendText(`${executable} compile "${editor.document.uri.fsPath}" "${outputPath}"`);
}

async function checkTBProgram() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) return;

    await editor.document.save();

    const config = vscode.workspace.getConfiguration('tb');
    const executable = config.get<string>('executablePath') || 'tb';

    try {
        const { stdout, stderr } = await execAsync(
            `${executable} check "${editor.document.uri.fsPath}"`
        );

        if (stdout) {
            vscode.window.showInformationMessage('✓ Check passed: ' + stdout);
        }
        if (stderr) {
            vscode.window.showErrorMessage('✗ Check failed: ' + stderr);
        }
    } catch (error: any) {
        vscode.window.showErrorMessage('Check failed: ' + error.message);
    }
}

class TBHoverProvider implements vscode.HoverProvider {
    provideHover(
        document: vscode.TextDocument,
        position: vscode.Position
    ): vscode.ProviderResult<vscode.Hover> {
        const range = document.getWordRangeAtPosition(position);
        const word = document.getText(range);

        const keywords: { [key: string]: string } = {
            'fn': 'Function definition',
            'let': 'Variable declaration (immutable)',
            'mut': 'Mutable variable modifier',
            'if': 'Conditional expression',
            'match': 'Pattern matching',
            'for': 'For loop',
            'async': 'Asynchronous function',
            'await': 'Await async result',
            'parallel': 'Parallel execution block'
        };

        if (keywords[word]) {
            return new vscode.Hover(keywords[word]);
        }
    }
}

class TBCompletionProvider implements vscode.CompletionItemProvider {
    provideCompletionItems(
        document: vscode.TextDocument,
        position: vscode.Position
    ): vscode.ProviderResult<vscode.CompletionItem[]> {
        const items: vscode.CompletionItem[] = [];

        // Keywords
        const keywords = ['fn', 'let', 'mut', 'if', 'else', 'match', 'for', 'while',
                         'async', 'await', 'parallel', 'return', 'break', 'continue'];

        keywords.forEach(kw => {
            const item = new vscode.CompletionItem(kw, vscode.CompletionItemKind.Keyword);
            items.push(item);
        });

        // Types
        const types = ['int', 'float', 'bool', 'string', 'list', 'dict'];
        types.forEach(t => {
            const item = new vscode.CompletionItem(t, vscode.CompletionItemKind.TypeParameter);
            items.push(item);
        });

        return items;
    }
}

export function deactivate() {}
