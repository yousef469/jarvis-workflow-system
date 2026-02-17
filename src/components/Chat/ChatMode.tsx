import { useState, useRef, useEffect, useCallback } from 'react'
import { useAIStore, type ChatMessage } from '../../stores/aiStore'
import socketService from '../../services/socket'

interface AttachedFile {
    name: string
    path: string
    content: string
    size: number
}

const suggestions = [
    "Explain quantum entanglement",
    "Write a Python sorting algorithm",
    "Help me debug my code",
    "Generate a REST API template",
    "Summarize this research paper",
    "Design a drone control system",
]

export default function ChatMode() {
    const { chatMessages, chatLoading, addChatMessage, updateLastChatMessage, setChatLoading, clearChat } = useAIStore()
    const [input, setInput] = useState('')
    const [attachedFiles, setAttachedFiles] = useState<AttachedFile[]>([])
    const [dragOver, setDragOver] = useState(false)
    const [jarvisStatus, setJarvisStatus] = useState<'idle' | 'listening' | 'thinking' | 'speaking'>('idle')
    const messagesEndRef = useRef<HTMLDivElement>(null)
    const textareaRef = useRef<HTMLTextAreaElement>(null)
    const dropzoneRef = useRef<HTMLDivElement>(null)

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [chatMessages])

    useEffect(() => {
        // Connect socket and listen for status updates
        socketService.connect()

        socketService.on('status_update', (data: any) => {
            console.log('[Socket] Jarvis Status:', data)
            if (data.status) {
                setJarvisStatus(data.status)
            }
        })

        socketService.on('transcription', (data: any) => {
            console.log('[Socket] Received transcription:', data)
            if (data.text) {
                // Add transcribed text to the chat immediately
                const userMsg: ChatMessage = { role: 'user', content: data.text, timestamp: Date.now() }
                addChatMessage(userMsg)
                setChatLoading(true)
            }
        })

        socketService.on('assistant_response', (data: any) => {
            console.log('[Socket] Assistant response:', data)
            setChatLoading(false)

            let imageUrl = undefined
            if (data.worker_result?.image_path) {
                // Convert absolute path to atlas-app:// protocol
                imageUrl = `atlas-app://${data.worker_result.image_path}`
            }

            addChatMessage({
                role: 'assistant',
                content: data.text || '',
                timestamp: Date.now(),
                image: imageUrl
            })
        })
    }, [])

    // ── Attach files via native dialog ──────────────────────────────
    const handleAttachFile = async () => {
        const filePath = await window.jarvis.dialog.openFile()
        if (!filePath) return
        await readAndAttach(filePath)
    }

    // ── Attach an entire folder (reads all readable files) ─────────
    const handleAttachFolder = async () => {
        const folderPath = await window.jarvis.dialog.openFolder()
        if (!folderPath) return
        await readFolder(folderPath)
    }

    const readFolder = async (dirPath: string, maxDepth = 3, currentDepth = 0) => {
        if (currentDepth >= maxDepth) return
        try {
            const entries = await window.jarvis.fs.readDirectory(dirPath)
            for (const entry of entries) {
                // Skip hidden files, node_modules, etc.
                if (entry.name.startsWith('.') || entry.name === 'node_modules' || entry.name === '__pycache__' || entry.name === '.git') continue

                if (entry.isFile) {
                    // Only read text-like files
                    const ext = entry.name.split('.').pop()?.toLowerCase() || ''
                    const textExts = ['py', 'js', 'ts', 'tsx', 'jsx', 'html', 'css', 'scss', 'json', 'md', 'txt', 'yaml', 'yml', 'toml', 'cfg', 'ini', 'sh', 'bash', 'c', 'cpp', 'h', 'hpp', 'java', 'rs', 'go', 'rb', 'php', 'xml', 'sql', 'csv', 'env', 'gitignore', 'dockerfile', 'makefile']
                    if (textExts.includes(ext) || entry.name.toLowerCase() === 'readme' || entry.name.toLowerCase() === 'makefile' || entry.name.toLowerCase() === 'dockerfile') {
                        await readAndAttach(entry.path)
                    }
                } else if (entry.isDirectory) {
                    await readFolder(entry.path, maxDepth, currentDepth + 1)
                }
            }
        } catch (err) {
            console.error('Failed to read folder:', err)
        }
    }

    const readAndAttach = async (filePath: string) => {
        // Don't re-attach same file
        if (attachedFiles.some(f => f.path === filePath)) return
        try {
            const content = await window.jarvis.fs.readFile(filePath)
            const name = filePath.split('/').pop() || filePath
            setAttachedFiles(prev => [...prev, { name, path: filePath, content, size: content.length }])
        } catch (err) {
            console.error('Failed to read file:', err)
        }
    }

    const removeFile = (path: string) => {
        setAttachedFiles(prev => prev.filter(f => f.path !== path))
    }

    // ── Drag & Drop support ────────────────────────────────────────
    const handleDragOver = useCallback((e: React.DragEvent) => {
        e.preventDefault()
        e.stopPropagation()
        setDragOver(true)
    }, [])

    const handleDragLeave = useCallback((e: React.DragEvent) => {
        e.preventDefault()
        e.stopPropagation()
        setDragOver(false)
    }, [])

    const handleDrop = useCallback(async (e: React.DragEvent) => {
        e.preventDefault()
        e.stopPropagation()
        setDragOver(false)

        // Electron provides file paths via dataTransfer.files
        const files = e.dataTransfer.files
        for (let i = 0; i < files.length; i++) {
            const file = files[i] as any
            const filePath = file.path
            if (filePath) {
                // Check if it's a directory or file
                try {
                    const stat = await window.jarvis.fs.stat(filePath)
                    if (stat.isDirectory) {
                        await readFolder(filePath)
                    } else {
                        await readAndAttach(filePath)
                    }
                } catch {
                    await readAndAttach(filePath)
                }
            }
        }
    }, [attachedFiles])

    // ── Build context with attached files ───────────────────────────
    const buildSystemPrompt = (): string => {
        if (attachedFiles.length === 0) return ''

        let context = `The user has shared ${attachedFiles.length} file(s) with you. Here are their contents:\n\n`
        for (const file of attachedFiles) {
            const truncated = file.content.length > 8000 ? file.content.slice(0, 8000) + '\n... [truncated]' : file.content
            context += `━━━ FILE: ${file.name} (${file.path}) ━━━\n${truncated}\n\n`
        }
        context += `You have full access to these files. Answer questions about them directly — explain the project, analyze the code, find bugs, suggest improvements, etc. Be specific and reference actual code from the files.`
        return context
    }

    // ── Send message ───────────────────────────────────────────────
    const sendMessage = async () => {
        const text = input.trim()
        if (!text || chatLoading) return

        // Show what files are attached in the user message
        let displayText = text
        if (attachedFiles.length > 0) {
            const fileList = attachedFiles.map(f => f.name).join(', ')
            displayText = `📎 [${fileList}]\n\n${text}`
        }

        // ── BUILD messages for AI BEFORE adding to state (avoids duplication) ──
        const fileContext = buildSystemPrompt()
        const messagesForAI: { role: string; content: string }[] = []

        // System prompt: always include anti-repetition instruction and force identity
        const systemContent = [
            "You are JARVIS, a loyal and professional AI assistant powered by the Qwen 2.5 Coder (7B) offline model. You are running LOCALLY on the user's Mac.",
            "NEVER call yourself GPT-4 or OpenAI. You are JARVIS.",
            "Be concise and clear. NEVER repeat words or phrases. Each sentence should add new information.",
            fileContext
        ].filter(Boolean).join('\n\n')
        messagesForAI.push({ role: 'system', content: systemContent })

        // Add EXISTING chat history (before we add the new user message)
        for (const m of chatMessages) {
            messagesForAI.push({ role: m.role, content: m.content })
        }

        // Add the user's NEW message (only once!)
        messagesForAI.push({ role: 'user', content: text })

        // NOW add to UI state
        const userMsg: ChatMessage = { role: 'user', content: displayText, timestamp: Date.now() }
        addChatMessage(userMsg)
        setInput('')
        setChatLoading(true)

        if (textareaRef.current) textareaRef.current.style.height = '24px'

        let cleanup: (() => void) | undefined
        try {
            // Add empty assistant message for streaming
            addChatMessage({ role: 'assistant', content: "", timestamp: Date.now() })

            cleanup = window.jarvis.ai.chatStream(messagesForAI, (chunk: string) => {
                if (chunk === '[DONE]') {
                    setChatLoading(false)
                    if (cleanup) cleanup()
                    return
                }
                updateLastChatMessage(chunk)
            })
        } catch (err) {
            addChatMessage({
                role: 'assistant',
                content: 'Failed to connect. Make sure Ollama is running.',
                timestamp: Date.now()
            })
            setChatLoading(false)
        } finally {
            // IPC listeners are internally managed but we should ensure clean state
            // In this specific implementation, chatStream returns a cleanup function
            // that is meant to be called when the stream is done or component unmounts.
            // However, since this is a one-off stream, we should call it once [DONE] is received
            // or if an error occurs.
        }
    }

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            sendMessage()
        }
    }

    const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
        setInput(e.target.value)
        const textarea = e.target
        textarea.style.height = '24px'
        textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px'
    }

    const handleVoiceClick = () => {
        console.log("Mic button clicked");
        if (jarvisStatus === 'listening') {
            // If already listening, we could stop it, but backend usually handles timeout
            return
        }
        if (!socketService.isConnected()) {
            console.error("Socket not connected! Attempting to reconnect...");
            socketService.connect();
        }
        console.log("Emitting trigger_voice event...");
        socketService.emit('trigger_voice', { timestamp: Date.now() });
    };

    return (
        <div
            className={`chat-container ${dragOver ? 'drag-over' : ''} ${jarvisStatus === 'listening' ? 'listening' : ''}`}
            ref={dropzoneRef}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
        >
            {/* Drag overlay */}
            {dragOver && (
                <div className="chat-drop-overlay">
                    <div className="chat-drop-content">
                        <span style={{ fontSize: 40 }}>📂</span>
                        <p>Drop files or folders here</p>
                        <p style={{ fontSize: 11, opacity: 0.6 }}>Jarvis will read them automatically</p>
                    </div>
                </div>
            )}

            {chatMessages.length === 0 ? (
                <div className="chat-empty">
                    <div className="chat-empty-icon">◈</div>
                    <h3>Jarvis AI Assistant</h3>
                    <p>
                        Your offline AI assistant for engineering, research, and creation.
                        <br />
                        <strong>Drop files or folders here</strong> — Jarvis reads them instantly.
                    </p>
                    <div className="chat-suggestions">
                        {suggestions.map((s, i) => (
                            <button key={i} className="chat-suggestion" onClick={() => { setInput(s); textareaRef.current?.focus() }}>
                                {s}
                            </button>
                        ))}
                    </div>
                </div>
            ) : (
                <div className="chat-messages">
                    {chatMessages.map((msg, i) => (
                        <div key={i} className={`chat-message ${msg.role}`}>
                            <div className="chat-avatar">
                                {msg.role === 'assistant' ? '◈' : '👤'}
                            </div>
                            <div className="chat-bubble">
                                {msg.image && (
                                    <div className="chat-image-attachment">
                                        <img src={msg.image} alt="Generated asset" />
                                    </div>
                                )}
                                {msg.content.split('\n').map((line: string, j: number) => (
                                    <p key={j}>{line}</p>
                                ))}
                            </div>
                        </div>
                    ))}
                    {(chatLoading || jarvisStatus !== 'idle') && (
                        <div className={`chat-message assistant ${jarvisStatus}`}>
                            <div className="chat-avatar">◈</div>
                            <div className="chat-bubble">
                                <div className="thinking-indicator">
                                    {jarvisStatus === 'listening' ? (
                                        <div className="listening-pulse">
                                            <span></span><span></span><span></span>
                                        </div>
                                    ) : (
                                        <><span></span><span></span><span></span></>
                                    )}
                                </div>
                                {jarvisStatus !== 'idle' && (
                                    <div className="status-text" style={{ fontSize: 10, opacity: 0.6, marginTop: 4 }}>
                                        {jarvisStatus.toUpperCase()}...
                                    </div>
                                )}
                            </div>
                        </div>
                    )}
                    <div ref={messagesEndRef} />
                </div>
            )}

            {/* Attached Files Bar */}
            {attachedFiles.length > 0 && (
                <div className="chat-attached-files">
                    {attachedFiles.map(file => (
                        <div key={file.path} className="attached-file-chip">
                            <span className="chip-icon">📄</span>
                            <span className="chip-name" title={file.path}>{file.name}</span>
                            <span className="chip-size">{file.size > 1024 ? `${Math.round(file.size / 1024)}KB` : `${file.size}B`}</span>
                            <button className="chip-remove" onClick={() => removeFile(file.path)}>✕</button>
                        </div>
                    ))}
                    <button className="clear-all-btn" onClick={() => setAttachedFiles([])}>Clear all</button>
                </div>
            )}

            {/* Input Area */}
            <div className="chat-input-area">
                <div className="chat-input-wrapper">
                    <div className="chat-actions">
                        <button className="chat-action-btn" onClick={handleAttachFile} title="Attach file">📎</button>
                        <button className="chat-action-btn" onClick={handleAttachFolder} title="Attach folder">📂</button>
                        <button
                            className={`chat-action-btn ${jarvisStatus === 'listening' ? 'active' : ''}`}
                            onClick={handleVoiceClick}
                            title="Voice input"
                        >
                            🎤
                        </button>
                    </div>
                    <textarea
                        ref={textareaRef}
                        value={input}
                        onChange={handleInput}
                        onKeyDown={handleKeyDown}
                        placeholder={attachedFiles.length > 0 ? `Ask about ${attachedFiles.length} attached file(s)...` : "Ask Jarvis anything..."}
                        rows={1}
                    />
                    <button
                        className="chat-send-btn"
                        onClick={sendMessage}
                        disabled={!input.trim() || chatLoading}
                    >
                        ↑
                    </button>
                </div>
            </div>
        </div>
    )
}
