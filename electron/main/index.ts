import { app, shell, BrowserWindow, ipcMain, protocol, net } from 'electron'
import { join } from 'path'
import { electronApp, optimizer, is } from '@electron-toolkit/utils'
import * as fs from 'fs'
import * as path from 'path'
import * as os from 'os'
import { spawn, ChildProcess } from 'child_process'
import * as pty from 'node-pty'

// Register scheme as privileged
protocol.registerSchemesAsPrivileged([
    { scheme: 'atlas-app', privileges: { standard: true, secure: true, supportFetchAPI: true, bypassCSP: true, stream: true } }
])

let mainWindow: BrowserWindow | null = null
let pythonBackend: ChildProcess | null = null

// ============================================================================
// GPU CONFIG — WebGL rendering setup
// ============================================================================
// This Mac has EGL driver issues (eglQueryDeviceAttribEXT crash).
// Use SwiftShader as primary renderer with performance optimizations.
app.commandLine.appendSwitch('enable-unsafe-swiftshader');
app.commandLine.appendSwitch('ignore-gpu-blocklist');
app.commandLine.appendSwitch('disable-gpu-compositing');  // Reduce GPU overhead
app.commandLine.appendSwitch('disable-gpu-vsync');         // Don't wait for vsync

// ============================================================================
// ATLAS PYTHON BACKEND LAUNCHER
// ============================================================================
function startPythonBackend(): void {
    const backendDir = is.dev
        ? path.join(process.cwd(), 'backend')
        : path.join(process.resourcesPath, 'backend')

    const serverScript = path.join(backendDir, 'server.py')

    if (!fs.existsSync(serverScript)) {
        console.warn(`[Atlas] Python backend not found at: ${serverScript}`)
        console.warn(`[Atlas] Running in frontend-only mode. Start backend manually.`)
        return
    }

    // Try common Python paths (Integrated Atlas Environment)
    const pythonPaths = [
        path.join(process.cwd(), 'atlas_env_mac', 'bin', 'python3'),
        path.join(process.cwd(), 'atlas_env', 'bin', 'python3'),
        'python3',
        'python'
    ]

    let pythonPath = 'python3'
    for (const p of pythonPaths) {
        if (fs.existsSync(p)) {
            pythonPath = p
            break
        }
    }

    console.log(`[Atlas] Checking if backend is already running on http://localhost:8765...`)

    // Quick check: if we started with 'start-all' or manually, don't spawn
    // We can use a simple fetch or net check
    import('net').then(net => {
        const socket = net.connect(8765, '127.0.0.1', () => {
            console.log(`[Atlas] External backend detected. Skipping internal launch.`)
            socket.destroy()
        })
        socket.on('error', () => {
            // No backend running, start it
            spawnBackend(pythonPath, serverScript, backendDir)
        })
    })
}

function spawnBackend(pythonPath: string, serverScript: string, backendDir: string): void {
    console.log(`[Atlas] Starting Python backend: "${pythonPath}" "${serverScript}"`)

    pythonBackend = spawn(pythonPath, [serverScript], {
        cwd: backendDir,
        env: {
            ...process.env,
            PYTHONUNBUFFERED: '1',
            KMP_DUPLICATE_LIB_OK: 'TRUE',
            OMP_NUM_THREADS: '1'
        }
    })

    pythonBackend.stdout?.on('data', (data) => {
        const msg = data.toString().trim()
        if (msg) console.log(`[Backend] ${msg}`)
        mainWindow?.webContents.send('backend:log', msg)
    })

    pythonBackend.stderr?.on('data', (data) => {
        const msg = data.toString().trim()
        if (msg) console.error(`[Backend] ${msg}`)
    })

    pythonBackend.on('exit', (code) => {
        console.log(`[Atlas] Python backend exited with code ${code}`)
        pythonBackend = null
    })
}

// ============================================================================
// WINDOW CREATION
// ============================================================================
function createWindow(): void {
    mainWindow = new BrowserWindow({
        width: 1500,
        height: 950,
        minWidth: 1000,
        minHeight: 700,
        show: false,
        autoHideMenuBar: false,
        titleBarStyle: 'hiddenInset',
        trafficLightPosition: { x: 15, y: 15 },
        backgroundColor: '#0a0a12',
        webPreferences: {
            preload: join(__dirname, '../preload/index.js'),
            sandbox: false,
            contextIsolation: true,
            nodeIntegration: false,
            webSecurity: false // Required to load local .glb models via file://
        }
    })

    mainWindow.on('ready-to-show', () => {
        mainWindow?.show()
    })

    mainWindow.webContents.setWindowOpenHandler((details) => {
        shell.openExternal(details.url)
        return { action: 'deny' }
    })

    const devUrl = process.env['VITE_DEV_SERVER_URL'] || 'http://127.0.0.1:5173'

    // Enable Screen Capture Support (getDisplayMedia)
    mainWindow.webContents.session.setDisplayMediaRequestHandler((_request, callback) => {
        const { desktopCapturer } = require('electron')
        console.log('[Electron] Requesting sources for getDisplayMedia...')

        desktopCapturer.getSources({ types: ['screen', 'window'] })
            .then((sources: any[]) => {
                if (sources.length > 0) {
                    console.log(`[Electron] Providing source: ${sources[0].name}`)
                    callback({ video: sources[0], audio: 'loopback' })
                } else {
                    console.warn('[Electron] No sources found for screen capture.')
                    callback({})
                }
            })
            .catch((err: any) => {
                console.error('[Electron] Error getting sources:', err)
                callback({})
            })
    })

    if (is.dev) {
        console.log(`[Atlas] Loading dev server: ${devUrl}`)
        mainWindow.loadURL(devUrl)
    } else {
        const prodPath = join(__dirname, '../../dist/index.html')
        mainWindow.loadFile(prodPath)
    }

    // ============================================================================
    // CRASH HANDLERS
    // ============================================================================
    mainWindow.webContents.on('render-process-gone', (event, details) => {
        console.error(`[Atlas] Render process gone: ${details.reason} (${details.exitCode})`)
        if (details.reason === 'crashed') {
            app.relaunch()
            app.exit(0)
        }
    })

    app.on('child-process-gone', (event, details) => {
        console.error(`[Atlas] Child process gone: ${details.type} - ${details.reason}`)
    })
}

// ============================================================================
// FILE SYSTEM IPC HANDLERS
// ============================================================================
ipcMain.handle('fs:readDirectory', async (_, dirPath: string) => {
    try {
        const entries = await fs.promises.readdir(dirPath, { withFileTypes: true })
        return entries.map(entry => ({
            name: entry.name,
            path: path.join(dirPath, entry.name),
            isDirectory: entry.isDirectory(),
            isFile: entry.isFile()
        }))
    } catch (error) {
        throw new Error(`Failed to read directory: ${error}`)
    }
})

ipcMain.handle('fs:readFile', async (_, filePath: string) => {
    try {
        return await fs.promises.readFile(filePath, 'utf-8')
    } catch (error) {
        throw new Error(`Failed to read file: ${error}`)
    }
})

ipcMain.handle('fs:writeFile', async (_, filePath: string, content: string) => {
    try {
        await fs.promises.writeFile(filePath, content, 'utf-8')
        return true
    } catch (error) {
        throw new Error(`Failed to write file: ${error}`)
    }
})

ipcMain.handle('fs:createFile', async (_, filePath: string) => {
    try {
        await fs.promises.writeFile(filePath, '', 'utf-8')
        return true
    } catch (error) {
        throw new Error(`Failed to create file: ${error}`)
    }
})

ipcMain.handle('fs:createDirectory', async (_, dirPath: string) => {
    try {
        await fs.promises.mkdir(dirPath, { recursive: true })
        return true
    } catch (error) {
        throw new Error(`Failed to create directory: ${error}`)
    }
})

ipcMain.handle('fs:delete', async (_, targetPath: string) => {
    try {
        const stat = await fs.promises.stat(targetPath)
        if (stat.isDirectory()) {
            await fs.promises.rm(targetPath, { recursive: true, force: true })
        } else {
            await fs.promises.unlink(targetPath)
        }
        return true
    } catch (error) {
        throw new Error(`Failed to delete: ${error}`)
    }
})

ipcMain.handle('fs:rename', async (_, oldPath: string, newPath: string) => {
    try {
        await fs.promises.rename(oldPath, newPath)
        return true
    } catch (error) {
        throw new Error(`Failed to rename: ${error}`)
    }
})

ipcMain.handle('fs:exists', async (_, targetPath: string) => {
    try {
        await fs.promises.access(targetPath)
        return true
    } catch {
        return false
    }
})

ipcMain.handle('fs:stat', async (_, targetPath: string) => {
    try {
        const stat = await fs.promises.stat(targetPath)
        return {
            isDirectory: stat.isDirectory(),
            isFile: stat.isFile(),
            size: stat.size,
            modified: stat.mtime.getTime(),
            created: stat.birthtime.getTime()
        }
    } catch (error) {
        throw new Error(`Failed to stat: ${error}`)
    }
})

// ============================================================================
// DIALOG IPC HANDLERS
// ============================================================================
ipcMain.handle('dialog:openFolder', async () => {
    const { dialog } = await import('electron')
    const result = await dialog.showOpenDialog(mainWindow!, {
        properties: ['openDirectory']
    })
    return result.canceled ? null : result.filePaths[0]
})

ipcMain.handle('dialog:openFile', async () => {
    const { dialog } = await import('electron')
    const win = BrowserWindow.getFocusedWindow() || mainWindow
    console.log(`[Dialog] Opening file dialog for window: ${!!win}`)

    const result = await dialog.showOpenDialog(win!, {
        properties: ['openFile'],
        filters: [
            { name: 'CAD/3D', extensions: ['glb', 'gltf', 'stl', 'obj', 'step', 'stp', 'iges', 'igs'] },
            { name: 'Documents', extensions: ['pdf', 'docx', 'pptx', 'xlsx', 'txt', 'md'] },
            { name: 'Code', extensions: ['py', 'js', 'ts', 'cpp', 'c', 'h', 'java', 'rs'] },
            { name: 'Images', extensions: ['png', 'jpg', 'jpeg', 'gif', 'svg'] },
            { name: 'All Files', extensions: ['*'] }
        ]
    })

    if (result.canceled) {
        console.log('[Dialog] User canceled.')
        return null
    }

    console.log(`[Dialog] User selected: ${result.filePaths[0]}`)
    return result.filePaths[0]
})

// ============================================================================
// TERMINAL IPC HANDLERS (node-pty)
// ============================================================================
const terminals: Map<string, pty.IPty> = new Map()

ipcMain.handle('terminal:create', async (_, id: string, cwd: string) => {
    try {
        const shellPath = process.platform === 'darwin' ? '/bin/zsh' : (process.env.SHELL || '/bin/bash')

        let finalCwd = cwd
        if (!cwd || !fs.existsSync(cwd)) {
            finalCwd = process.env.HOME || os.homedir() || '/'
        }

        const ptyProcess = pty.spawn(shellPath, ['-l'], {
            name: 'xterm-256color',
            cols: 80,
            rows: 24,
            cwd: finalCwd,
            env: {
                ...process.env,
                PATH: process.env.PATH || '/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin',
                TERM: 'xterm-256color',
                COLORTERM: 'truecolor'
            } as Record<string, string>
        })

        terminals.set(id, ptyProcess)

        ptyProcess.onData((data) => {
            mainWindow?.webContents.send('terminal:data', id, data)
        })

        ptyProcess.onExit(({ exitCode }) => {
            mainWindow?.webContents.send('terminal:exit', id, exitCode)
            terminals.delete(id)
        })

        return id
    } catch (error: any) {
        console.error('[Atlas] Failed to create terminal:', error)
        throw error
    }
})

ipcMain.handle('terminal:write', async (_, id: string, data: string) => {
    const ptyProcess = terminals.get(id)
    if (ptyProcess) ptyProcess.write(data)
})

ipcMain.handle('terminal:kill', async (_, id: string) => {
    const ptyProcess = terminals.get(id)
    if (ptyProcess) {
        ptyProcess.kill()
        terminals.delete(id)
    }
})

ipcMain.handle('terminal:resize', async (_, id: string, cols: number, rows: number) => {
    const ptyProcess = terminals.get(id)
    if (ptyProcess) ptyProcess.resize(cols, rows)
})

// ============================================================================
// AI / LLM IPC HANDLERS (Ollama)
// ============================================================================
ipcMain.handle('ai:chat', async (_, messages: unknown[]) => {
    try {
        const controller = new AbortController()
        const timeout = setTimeout(() => controller.abort(), 300000) // 5 minute timeout for extreme coder tasks

        const response = await fetch('http://localhost:11434/api/chat', {
            method: 'POST',
            body: JSON.stringify({
                model: 'qwen2.5-coder:7b',
                messages: messages,
                stream: false,
                options: {
                    temperature: 0.7,
                    repeat_penalty: 1.2,
                    repeat_last_n: 128,
                    num_predict: 512,
                    num_ctx: 4096,
                    top_k: 40,
                    top_p: 0.9
                }
            }),
            headers: { 'Content-Type': 'application/json' },
            signal: controller.signal
        })
        clearTimeout(timeout)

        if (!response.ok) {
            const errorText = await response.text()
            throw new Error(`Ollama API error: ${response.status} - ${errorText}`)
        }

        const data = await response.json() as any
        return data.message.content
    } catch (error: any) {
        console.error('[Atlas] AI Chat Error:', error)
        if (error.name === 'AbortError') {
            return "Error: AI reasoning timed out. The workspace context might be too large for the current model."
        }
        return `Failed to connect to Ollama: ${error.message}. Ensure Ollama is running and 'qwen2.5-coder:7b' is installed.`
    }
})

ipcMain.handle('ai:chat-stream', async (event, messages: unknown[]) => {
    try {
        const response = await fetch('http://localhost:11434/api/chat', {
            method: 'POST',
            body: JSON.stringify({
                model: 'qwen2.5-coder:7b',
                messages: messages,
                stream: true,
                options: {
                    temperature: 0.7,
                    repeat_penalty: 1.2,
                    repeat_last_n: 128,
                    num_predict: 512,
                    num_ctx: 4096,
                    top_k: 40,
                    top_p: 0.9
                }
            }),
            headers: { 'Content-Type': 'application/json' }
        })

        if (!response.ok) {
            event.sender.send('ai:chat-chunk', `Error: Ollama API error (${response.status})`)
            return
        }

        const body = response.body
        if (!body) {
            event.sender.send('ai:chat-chunk', 'Error: Response body is null.')
            return
        }

        const reader = body.getReader()
        const decoder = new TextDecoder()
        let buffer = ''

        while (true) {
            const { done, value } = await reader.read()
            if (done) break

            buffer += decoder.decode(value, { stream: true })
            const lines = buffer.split('\n')

            // Keep the last (potentially incomplete) line in the buffer
            buffer = lines.pop() || ''

            for (const line of lines) {
                if (!line.trim()) continue
                try {
                    const data = JSON.parse(line)
                    if (data.message?.content) {
                        event.sender.send('ai:chat-chunk', data.message.content)
                    }
                    if (data.done) {
                        event.sender.send('ai:chat-chunk', '[DONE]')
                    }
                } catch (e) {
                    // This should be rare now with the buffer
                }
            }
        }
    } catch (error: any) {
        event.sender.send('ai:chat-chunk', `Error: ${error.message}`)
    }
})

ipcMain.handle('ai:coder-stream', async (event, payload: { text: string, project_path: string, active_file?: string }) => {
    try {
        console.log('[Atlas] Proxying Coder Request to Python Backend...')
        const response = await fetch('http://localhost:8765/api/coder/stream', {
            method: 'POST',
            body: JSON.stringify(payload),
            headers: { 'Content-Type': 'application/json' }
        })

        if (!response.ok) {
            const err = await response.text()
            event.sender.send('ai:coder-chunk', `Error: Backend error (${response.status}) - ${err}`)
            return
        }

        const body = response.body
        if (!body) {
            event.sender.send('ai:coder-chunk', 'Error: Response body is null.')
            return
        }

        const reader = body.getReader()
        const decoder = new TextDecoder()
        let buffer = ''

        while (true) {
            const { done, value } = await reader.read()
            if (done) break

            buffer += decoder.decode(value, { stream: true })
            const lines = buffer.split('\n')

            // Keep the last (potentially incomplete) line in the buffer
            buffer = lines.pop() || ''

            for (const line of lines) {
                if (!line.trim()) continue
                // line is a JSON string of Partial<CoderResponse>
                event.sender.send('ai:coder-chunk', line)
            }
        }
        event.sender.send('ai:coder-chunk', '[DONE]')

    } catch (error: any) {
        console.error('[Atlas] Coder Bridge Error:', error)
        event.sender.send('ai:coder-chunk', `Error: ${error.message}. Is the Python backend running?`)
    }
})

ipcMain.handle('ai:explain', async (_, code: string) => {
    try {
        const response = await fetch('http://localhost:11434/api/generate', {
            method: 'POST',
            body: JSON.stringify({
                model: 'qwen2.5-coder:7b',
                prompt: `Explain this code briefly and suggest improvements:\n\n\`\`\`\n${code}\n\`\`\``,
                stream: false
            }),
            headers: { 'Content-Type': 'application/json' }
        })
        const data = await response.json() as any
        return data.response
    } catch (error) {
        return "Failed to connect to Ollama."
    }
})

ipcMain.handle('ai:analyzeCode', async (_, filePath: string) => {
    const ext = path.extname(filePath).toLowerCase()
    const workerDir = path.join(process.cwd(), 'workers')

    const runWorker = (cmd: string, args: string[]): Promise<unknown> => {
        return new Promise((resolve, reject) => {
            const proc = spawn(cmd, args)
            let output = ''
            proc.stdout.on('data', (data) => { output += data.toString() })
            proc.on('close', () => {
                try {
                    resolve(JSON.parse(output))
                } catch {
                    reject(`Failed to parse worker output: ${output}`)
                }
            })
        })
    }

    if (['.py'].includes(ext)) {
        const venvPython = path.join(workerDir, 'python', 'venv', 'bin', 'python3')
        return runWorker(venvPython, [path.join(workerDir, 'python', 'worker.py'), filePath])
    } else if (['.js', '.jsx', '.ts', '.tsx'].includes(ext)) {
        return runWorker('node', [path.join(workerDir, 'js', 'worker.js'), filePath])
    } else if (['.html'].includes(ext)) {
        return runWorker('node', [path.join(workerDir, 'html', 'worker.js'), filePath])
    } else if (['.css'].includes(ext)) {
        return runWorker('node', [path.join(workerDir, 'css', 'worker.js'), filePath])
    }

    return {
        complexity: 1,
        patterns: [],
        diagnostics: [],
        suggestions: [{ title: "Unsupported Language", description: "AI analysis coming soon for this file type." }]
    }
})

ipcMain.handle('atlas:getSources', async () => {
    try {
        const { desktopCapturer } = require('electron')
        const sources = await desktopCapturer.getSources({ types: ['window', 'screen'], thumbnailSize: { width: 150, height: 150 } })
        return sources.map((s: any) => ({
            id: s.id,
            name: s.name,
            thumbnail: s.thumbnail.toDataURL()
        }))
    } catch (error) {
        console.error('[Atlas] Failed to get sources:', error)
        return []
    }
})

// ============================================================================
// ATLAS-SPECIFIC IPC HANDLERS
// ============================================================================
ipcMain.handle('atlas:captureFrame', async () => {
    try {
        const { desktopCapturer, screen } = require('electron')
        const primaryDisplay = screen.getPrimaryDisplay()
        const { width, height } = primaryDisplay.size

        console.log(`[Atlas] Capturing primary display: ${width}x${height}`)

        const sources = await desktopCapturer.getSources({
            types: ['screen'],
            thumbnailSize: { width: Math.floor(width / 2), height: Math.floor(height / 2) }
        })

        let finalSources = sources
        if (finalSources.length === 0) {
            console.log('[Atlas] No screens found, trying window capture...')
            finalSources = await desktopCapturer.getSources({
                types: ['window'],
                thumbnailSize: { width: 800, height: 600 }
            })
        }

        if (finalSources.length > 0) {
            const dataUrl = finalSources[0].thumbnail.toDataURL()
            console.log(`[Atlas] Source found: ${finalSources[0].name}. Thumbnail size: ${dataUrl.length} chars`)
            return dataUrl
        }

        console.warn('[Atlas] No capture sources found in captureFrame.')
        return null
    } catch (error) {
        console.error('[Atlas] Direct capture failed:', error)
        return null
    }
})

ipcMain.handle('atlas:getBackendStatus', async () => {
    try {
        const response = await fetch('http://localhost:8765/api/health', { signal: AbortSignal.timeout(2000) })
        return response.ok ? 'connected' : 'error'
    } catch {
        return 'disconnected'
    }
})

ipcMain.handle('atlas:getSystemInfo', async () => {
    return {
        platform: process.platform,
        arch: process.arch,
        cpus: os.cpus().length,
        totalMemory: Math.round(os.totalmem() / (1024 * 1024 * 1024)),
        freeMemory: Math.round(os.freemem() / (1024 * 1024 * 1024)),
        homeDir: os.homedir()
    }
})

// ============================================================================
// APP LIFECYCLE
// ============================================================================
app.whenReady().then(() => {
    // ============================================================================
    // CUSTOM PROTOCOL HANDLER (Safe File Access)
    // ================================= scheme: atlas-app://path/to/file
    protocol.handle('atlas-app', async (request) => {
        // atlas-app:///Users/yousef/... => extract the path after atlas-app://
        // IMPORTANT: Do NOT use new URL() — it lowercases the hostname portion,
        // turning /Users/ into /users/ which breaks case-sensitive Mac paths.
        let filePath = decodeURIComponent(request.url.replace('atlas-app://', ''))

        // Ensure absolute path on Mac
        if (process.platform !== 'win32' && !filePath.startsWith('/')) {
            filePath = '/' + filePath
        }

        // Fix: URL sometimes strips leading slash, so /Users becomes Users
        // Also handle case where the url parser lowercased the first segment
        if (filePath.startsWith('/users/') && !fs.existsSync(filePath)) {
            filePath = '/Users/' + filePath.slice(7)
        }

        console.log(`[Protocol] Request: ${request.url} -> Resolved Path: ${filePath}`)

        try {
            if (!fs.existsSync(filePath)) {
                console.error(`[Protocol] File NOT FOUND: ${filePath}`)
                return new Response('File not found', { status: 404 })
            }

            const buffer = await fs.promises.readFile(filePath)
            const extension = path.extname(filePath).toLowerCase()

            let contentType = 'application/octet-stream'
            if (extension === '.glb') contentType = 'model/gltf-binary'
            if (extension === '.gltf') contentType = 'model/gltf+json'
            if (extension === '.stl') contentType = 'model/stl'

            return new Response(buffer, {
                status: 200,
                headers: {
                    'Content-Type': contentType,
                    'Access-Control-Allow-Origin': '*',
                    'Cache-Control': 'no-cache',
                    'Content-Length': buffer.length.toString()
                }
            })
        } catch (e) {
            console.error(`[Protocol] CRASH: ${e}`)
            return new Response('Internal error', { status: 500 })
        }
    })

    electronApp.setAppUserModelId('com.atlas.app')

    app.on('browser-window-created', (_, window) => {
        optimizer.watchWindowShortcuts(window)
    })

    createWindow()
    startPythonBackend()

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) createWindow()
    })
})

app.on('window-all-closed', () => {
    // Kill all terminals
    terminals.forEach(t => t.kill())
    terminals.clear()

    // Kill Python backend
    if (pythonBackend) {
        pythonBackend.kill()
        pythonBackend = null
    }

    if (process.platform !== 'darwin') {
        app.quit()
    }
})
