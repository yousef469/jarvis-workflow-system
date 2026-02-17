import { useState, useRef, useEffect, useCallback } from 'react'
import { useAIStore, type ChatMessage } from '../../stores/aiStore'
import { useFileStore, type FileEntry } from '../../stores/fileStore'

/**
 * Unified AI Chat Panel for Code Mode.
 * 
 * - Automatically reads ALL files when a folder is opened
 * - Provides incremental context updates (doesn't wait for full scan)
 * - Can write code changes back to files
 */

interface PanelMessage extends ChatMessage {
    isFileWrite?: boolean
}

export default function AIPanel() {
    const { toggleAiPanel } = useAIStore()
    const {
        workspacePath, fileTree, openFiles, activeFilePath,
        updateFileContent, setWorkspacePath, setFileTree
    } = useFileStore()

    const [messages, setMessages] = useState<PanelMessage[]>([])
    const [input, setInput] = useState('')
    const [loading, setLoading] = useState(false)
    const [workspaceContext, setWorkspaceContext] = useState('')
    const [contextLoaded, setContextLoaded] = useState(false)
    const [fileCount, setFileCount] = useState(0)
    const [isScanning, setIsScanning] = useState(false)
    const messagesEndRef = useRef<HTMLDivElement>(null)
    const inputRef = useRef<HTMLTextAreaElement>(null)
    const scanAbortRef = useRef<boolean>(false)

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [messages])

    // ── Debug Logging ──────────────────────────────────────────────
    const logContext = () => {
        console.group('Jarvis AI Context Debug')
        console.log('Workspace Path:', workspacePath)
        console.log('File Count:', fileCount)
        console.log('Scanning:', isScanning)
        console.log('Context Ready:', contextLoaded)
        console.log('Context Sample (first 500 chars):', workspaceContext.slice(0, 500))
        console.log('Active File:', activeFilePath)
        console.groupEnd()
        alert(`Debug: ${fileCount} files scanned. Check Console (F12) for details.`)
    }

    // ── Open Folder ───────────────────────────────────────────────
    const handleOpenFolder = async () => {
        const folderPath = await window.jarvis.dialog.openFolder()
        if (!folderPath) return
        setWorkspacePath(folderPath)
        try {
            const entries = await window.jarvis.fs.readDirectory(folderPath)
            const tree: FileEntry[] = entries.map((e: any) => ({
                name: e.name, path: e.path, isDirectory: e.isDirectory,
                isFile: e.isFile, children: e.isDirectory ? [] : undefined, isExpanded: false
            }))
            setFileTree(tree)
        } catch (err) { console.error('Failed to read folder:', err) }
    }

    // ── Load workspace context (Incremental) ──────────────────────
    const loadWorkspaceContext = useCallback(async () => {
        if (!workspacePath) {
            setWorkspaceContext('')
            setContextLoaded(false)
            setFileCount(0)
            return
        }

        setIsScanning(true)
        setContextLoaded(false)
        scanAbortRef.current = false

        let context = ''
        let count = 0
        const textExts = new Set([
            'py', 'js', 'ts', 'tsx', 'jsx', 'html', 'css', 'scss', 'json', 'md', 'txt',
            'yaml', 'yml', 'toml', 'cfg', 'ini', 'sh', 'bash', 'c', 'cpp', 'h', 'hpp',
            'java', 'rs', 'go', 'rb', 'php', 'xml', 'sql', 'csv', 'env'
        ])
        const skipDirs = new Set(['.git', 'node_modules', '__pycache__', 'dist', 'build', '.next', '.venv', 'venv', 'env', 'out', 'dist-electron'])

        const readDir = async (dirPath: string, depth: number) => {
            if (depth >= 4 || scanAbortRef.current || count >= 15) return // Stop at 15 files for snappy 7B reasoning
            try {
                const entries = await window.jarvis.fs.readDirectory(dirPath)
                for (const entry of entries) {
                    if (scanAbortRef.current) break
                    if (entry.name.startsWith('.') && entry.name !== '.env') continue
                    if (skipDirs.has(entry.name)) continue

                    if (entry.isFile) {
                        const ext = entry.name.split('.').pop()?.toLowerCase() || ''
                        const nameLC = entry.name.toLowerCase()
                        if (textExts.has(ext) || nameLC === 'readme' || nameLC === 'makefile' || nameLC === 'dockerfile') {
                            try {
                                const content = await window.jarvis.fs.readFile(entry.path)
                                // Truncate more aggressively for 1B models
                                const truncated = content.length > 1200
                                    ? content.slice(0, 1200) + '\n... [truncated]'
                                    : content
                                const relPath = entry.path.replace(workspacePath + '/', '')
                                context += `FILE: ${relPath}\n\`\`\`\n${truncated}\n\`\`\`\n\n`
                                count++

                                if (count % 10 === 0) {
                                    setWorkspaceContext(context)
                                    setFileCount(count)
                                }
                            } catch { /* skip */ }
                        }
                    } else if (entry.isDirectory) {
                        await readDir(entry.path, depth + 1)
                    }
                }
            } catch { /* skip */ }
        }

        await readDir(workspacePath, 0)
        if (!scanAbortRef.current) {
            setWorkspaceContext(context)
            setFileCount(count)
            setContextLoaded(true)
        }
        setIsScanning(false)
    }, [workspacePath])

    useEffect(() => {
        loadWorkspaceContext()
        return () => { scanAbortRef.current = true }
    }, [loadWorkspaceContext])

    // ── Send message ───────────────────────────────────────────────
    const sendMessage = async () => {
        const text = input.trim()
        if (!text || loading) return

        const userMsg: PanelMessage = { role: 'user', content: text, timestamp: Date.now() }
        setMessages(prev => [...prev, userMsg])
        setInput('')
        setLoading(true)

        if (inputRef.current) inputRef.current.style.height = '24px'

        // ── Jarvis Advanced Coder Engine (Streaming) ──
        try {
            console.log('[Jarvis] Starting Advanced Coder Stream...')

            // Create a pending message for the assistant
            setMessages(prev => [...prev, { role: 'assistant', content: 'Thinking...', timestamp: Date.now() }])
            let lastState: any = { thought_process: '', operations: [] }

            const payload = {
                text: text,
                project_path: workspacePath || '',
                active_file: activeFilePath || undefined
            }

            const unsub = window.jarvis.ai.coderStream(payload, (chunk: string) => {
                if (chunk === '[DONE]') {
                    setLoading(false)
                    processFileOperations(lastState.operations)
                    return
                }

                if (chunk.startsWith('Error:')) {
                    setMessages(prev => {
                        const last = prev[prev.length - 1]
                        return [...prev.slice(0, -1), { ...last, content: chunk }]
                    })
                    setLoading(false)
                    return
                }

                try {
                    const parsed = JSON.parse(chunk)
                    // Instructor Partial yields the FULL object state so far
                    lastState = parsed

                    setMessages(prev => {
                        const last = prev[prev.length - 1]
                        if (last?.role === 'assistant') {
                            return [...prev.slice(0, -1), { ...last, content: parsed.thought_process || '...' }]
                        }
                        return prev
                    })

                } catch (e) {
                    // Ignore JSON parse errors for partial chunks if any (should work line by line though)
                }
            })

            // Safety cleanup
            setTimeout(() => unsub(), 600000) // 10 minutes for heavy RAG

        } catch (err) {
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: 'Error: Connection lost. Ensure Python Backend is running.',
                timestamp: Date.now()
            }])
            setLoading(false)
        }
    }

    const processFileOperations = async (operations: any[]) => {
        if (!operations || operations.length === 0) return

        for (const op of operations) {
            try {
                const { operation, path, content, explanation } = op
                if (operation === 'create' || operation === 'modify') {
                    console.log(`[Jarvis] Writing file: ${path}`)
                    await window.jarvis.fs.writeFile(path, content || '')
                    // Update the file store if it's open
                    updateFileContent(path, content || '')
                } else if (operation === 'delete') {
                    console.log(`[Jarvis] Deleting file: ${path}`)
                    // await window.jarvis.fs.deleteFile(path) // If delete exists in API
                }
            } catch (err) { console.error('File Op failed:', err) }
        }
    }

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            sendMessage()
        }
    }

    return (
        <div className="ai-panel">
            <div className="ai-panel-header">
                <h3>🤖 Jarvis AI</h3>
                <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                    <button onClick={logContext} style={{ fontSize: 10, opacity: 0.5 }}>⚙️</button>
                    {workspacePath && (
                        <span
                            className={`context-badge ${contextLoaded ? 'loaded' : 'loading'}`}
                            onClick={() => loadWorkspaceContext()}
                            style={{ cursor: 'pointer' }}
                        >
                            {isScanning ? `⏳ (${fileCount})` : `📂 ${fileCount}`}
                        </span>
                    )}
                    <button onClick={toggleAiPanel}>✕</button>
                </div>
            </div>

            {workspaceContext.length > 50000 && contextLoaded && (
                <div className="context-warning">
                    ⚠️ Dense context ({Math.round(workspaceContext.length / 1024)}KB). Performance may vary.
                </div>
            )}

            <div className="ai-panel-content">
                {messages.length === 0 ? (
                    <div className="ai-panel-empty">
                        <div style={{ fontSize: 32, opacity: 0.2, marginBottom: 8 }}>◈</div>
                        {!workspacePath ? (
                            <>
                                <p style={{ color: 'var(--text-muted)', fontSize: 12, textAlign: 'center', lineHeight: 1.6, maxWidth: 220 }}>
                                    Open a project folder to start.
                                </p>
                                <button className="ai-open-folder-btn" onClick={handleOpenFolder}>
                                    📂 Open Folder
                                </button>
                            </>
                        ) : (
                            <>
                                <p style={{ color: 'var(--text-muted)', fontSize: 12, textAlign: 'center', lineHeight: 1.6, maxWidth: 220 }}>
                                    {isScanning ? 'Reading files...' : `Jarvis is ready with ${fileCount} files.`}
                                </p>
                                <div className="ai-quick-actions">
                                    <button onClick={() => setInput('What is this project about?')}>What is this project?</button>
                                    <button onClick={() => setInput('Find bugs in the code')}>Find bugs</button>
                                </div>
                            </>
                        )}
                    </div>
                ) : (
                    <div className="ai-messages">
                        {messages.map((msg, i) => (
                            <div key={i} className={`ai-msg ${msg.role}`}>
                                <div className="ai-msg-avatar">{msg.role === 'assistant' ? '◈' : '👤'}</div>
                                <div className="ai-msg-content">
                                    {msg.content.split('\n').map((line, j) => <p key={j}>{line || '\u00A0'}</p>)}
                                    {msg.isFileWrite && <div className="file-write-badge">✅ Files updated</div>}
                                </div>
                            </div>
                        ))}
                        {loading && (
                            <div className="ai-msg assistant">
                                <div className="ai-msg-avatar">◈</div>
                                <div className="ai-msg-content">
                                    <div className="thinking-indicator"><span></span><span></span><span></span></div>
                                </div>
                            </div>
                        )}
                        <div ref={messagesEndRef} />
                    </div>
                )}
            </div>

            <div className="ai-panel-input">
                <textarea
                    ref={inputRef}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Ask Jarvis..."
                    rows={1}
                />
                <button className="ai-send-btn" onClick={sendMessage} disabled={!input.trim() || loading}>↑</button>
            </div>
        </div>
    )
}
