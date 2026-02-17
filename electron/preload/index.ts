import { contextBridge, ipcRenderer } from 'electron'

export interface FileEntry {
    name: string
    path: string
    isDirectory: boolean
    isFile: boolean
}

export interface FileStat {
    isDirectory: boolean
    isFile: boolean
    size: number
    modified: number
    created: number
}

// Expose Jarvis API to renderer
const jarvisAPI = {
    // File System
    fs: {
        readDirectory: (dirPath: string): Promise<FileEntry[]> =>
            ipcRenderer.invoke('fs:readDirectory', dirPath),
        readFile: (filePath: string): Promise<string> =>
            ipcRenderer.invoke('fs:readFile', filePath),
        writeFile: (filePath: string, content: string): Promise<boolean> =>
            ipcRenderer.invoke('fs:writeFile', filePath, content),
        createFile: (filePath: string): Promise<boolean> =>
            ipcRenderer.invoke('fs:createFile', filePath),
        createDirectory: (dirPath: string): Promise<boolean> =>
            ipcRenderer.invoke('fs:createDirectory', dirPath),
        delete: (targetPath: string): Promise<boolean> =>
            ipcRenderer.invoke('fs:delete', targetPath),
        rename: (oldPath: string, newPath: string): Promise<boolean> =>
            ipcRenderer.invoke('fs:rename', oldPath, newPath),
        exists: (targetPath: string): Promise<boolean> =>
            ipcRenderer.invoke('fs:exists', targetPath),
        stat: (targetPath: string): Promise<FileStat> =>
            ipcRenderer.invoke('fs:stat', targetPath),
    },

    // Dialog
    dialog: {
        openFolder: (): Promise<string | null> =>
            ipcRenderer.invoke('dialog:openFolder'),
        openFile: (): Promise<string | null> =>
            ipcRenderer.invoke('dialog:openFile'),
    },

    // Terminal
    terminal: {
        create: (id: string, cwd: string): Promise<string> =>
            ipcRenderer.invoke('terminal:create', id, cwd),
        write: (id: string, data: string): Promise<void> =>
            ipcRenderer.invoke('terminal:write', id, data),
        kill: (id: string): Promise<void> =>
            ipcRenderer.invoke('terminal:kill', id),
        resize: (id: string, cols: number, rows: number): Promise<void> =>
            ipcRenderer.invoke('terminal:resize', id, cols, rows),
        onData: (callback: (id: string, data: string) => void) => {
            const handler = (_: any, id: string, data: string) => callback(id, data)
            ipcRenderer.on('terminal:data', handler)
            return () => ipcRenderer.removeListener('terminal:data', handler)
        },
        onExit: (callback: (id: string, exitCode: number) => void) => {
            const handler = (_: any, id: string, exitCode: number) => callback(id, exitCode)
            ipcRenderer.on('terminal:exit', handler)
            return () => ipcRenderer.removeListener('terminal:exit', handler)
        },
    },

    // AI
    ai: {
        chat: (messages: unknown[]): Promise<string> =>
            ipcRenderer.invoke('ai:chat', messages),
        chatStream: (messages: unknown[], callback: (chunk: string) => void) => {
            const handler = (_: any, chunk: string) => callback(chunk)
            ipcRenderer.removeAllListeners('ai:chat-chunk')
            ipcRenderer.on('ai:chat-chunk', handler)
            ipcRenderer.invoke('ai:chat-stream', messages)
            return () => ipcRenderer.removeListener('ai:chat-chunk', handler)
        },
        coderStream: (payload: { text: string, project_path: string, active_file?: string }, callback: (chunk: string) => void) => {
            const handler = (_: any, chunk: string) => callback(chunk)
            ipcRenderer.removeAllListeners('ai:coder-chunk')
            ipcRenderer.on('ai:coder-chunk', handler)
            ipcRenderer.invoke('ai:coder-stream', payload)
            return () => ipcRenderer.removeListener('ai:coder-chunk', handler)
        },
        explain: (code: string): Promise<string> =>
            ipcRenderer.invoke('ai:explain', code),
        analyzeCode: (filePath: string): Promise<unknown> =>
            ipcRenderer.invoke('ai:analyzeCode', filePath),
    },

    // Jarvis-specific
    captureFrame: (): Promise<string | null> =>
        ipcRenderer.invoke('jarvis:captureFrame'),
    getSources: (): Promise<any[]> =>
        ipcRenderer.invoke('jarvis:getSources'),
    getBackendStatus: (): Promise<string> =>
        ipcRenderer.invoke('jarvis:getBackendStatus'),
    getSystemInfo: (): Promise<any> =>
        ipcRenderer.invoke('jarvis:getSystemInfo'),

    // Backend logs
    onBackendLog: (callback: (msg: string) => void) => {
        const handler = (_: any, msg: string) => callback(msg)
        ipcRenderer.on('backend:log', handler)
        return () => ipcRenderer.removeListener('backend:log', handler)
    },
}

contextBridge.exposeInMainWorld('jarvis', jarvisAPI)
contextBridge.exposeInMainWorld('jarvis', jarvisAPI)
