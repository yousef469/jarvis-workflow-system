import { create } from 'zustand'

export interface FileEntry {
    name: string
    path: string
    isDirectory: boolean
    isFile: boolean
    children?: FileEntry[]
    isExpanded?: boolean
}

export interface OpenFile {
    path: string
    name: string
    content: string
    isDirty: boolean
}

interface FileStore {
    workspacePath: string | null
    fileTree: FileEntry[]
    openFiles: OpenFile[]
    activeFilePath: string | null

    setWorkspacePath: (path: string | null) => void
    setFileTree: (tree: FileEntry[]) => void
    toggleExpand: (path: string) => void
    openFile: (file: OpenFile) => void
    closeFile: (path: string) => void
    setActiveFile: (path: string) => void
    updateFileContent: (path: string, content: string) => void
    markFileSaved: (path: string) => void
}

declare global {
    interface Window {
        jarvis: {
            fs: {
                readDirectory: (path: string) => Promise<{ name: string; path: string; isDirectory: boolean; isFile: boolean }[]>
                readFile: (path: string) => Promise<string>
                writeFile: (path: string, content: string) => Promise<boolean>
                createFile: (path: string) => Promise<boolean>
                createDirectory: (path: string) => Promise<boolean>
                delete: (path: string) => Promise<boolean>
                rename: (old: string, newPath: string) => Promise<boolean>
                exists: (path: string) => Promise<boolean>
                stat: (path: string) => Promise<any>
            }
            dialog: {
                openFolder: () => Promise<string | null>
                openFile: () => Promise<string | null>
            }
            terminal: {
                create: (id: string, cwd: string) => Promise<string>
                write: (id: string, data: string) => Promise<void>
                kill: (id: string) => Promise<void>
                resize: (id: string, cols: number, rows: number) => Promise<void>
                onData: (cb: (id: string, data: string) => void) => () => void
                onExit: (cb: (id: string, code: number) => void) => () => void
            }
            ai: {
                chat: (messages: unknown[]) => Promise<string>
                chatStream: (messages: unknown[], callback: (chunk: string) => void) => () => void
                coderStream: (payload: { text: string, project_path: string, active_file?: string }, callback: (chunk: string) => void) => () => void
                explain: (code: string) => Promise<string>
                analyzeCode: (path: string) => Promise<any>
            }
            getBackendStatus: () => Promise<string>
            getSystemInfo: () => Promise<any>
            onBackendLog: (cb: (msg: string) => void) => () => void
        }
    }
}

export const useFileStore = create<FileStore>((set, get) => ({
    workspacePath: null,
    fileTree: [],
    openFiles: [],
    activeFilePath: null,

    setWorkspacePath: (path) => set({ workspacePath: path }),
    setFileTree: (tree) => set({ fileTree: tree }),

    toggleExpand: (path) => set((state) => {
        const toggle = (entries: FileEntry[]): FileEntry[] =>
            entries.map(e => {
                if (e.path === path) return { ...e, isExpanded: !e.isExpanded }
                if (e.children) return { ...e, children: toggle(e.children) }
                return e
            })
        return { fileTree: toggle(state.fileTree) }
    }),

    openFile: (file) => set((state) => {
        const exists = state.openFiles.find(f => f.path === file.path)
        if (exists) return { activeFilePath: file.path }
        return {
            openFiles: [...state.openFiles, file],
            activeFilePath: file.path
        }
    }),

    closeFile: (path) => set((state) => {
        const filtered = state.openFiles.filter(f => f.path !== path)
        const newActive = state.activeFilePath === path
            ? (filtered.length > 0 ? filtered[filtered.length - 1].path : null)
            : state.activeFilePath
        return { openFiles: filtered, activeFilePath: newActive }
    }),

    setActiveFile: (path) => set({ activeFilePath: path }),

    updateFileContent: (path, content) => set((state) => ({
        openFiles: state.openFiles.map(f =>
            f.path === path ? { ...f, content, isDirty: true } : f
        )
    })),

    markFileSaved: (path) => set((state) => ({
        openFiles: state.openFiles.map(f =>
            f.path === path ? { ...f, isDirty: false } : f
        )
    })),
}))
