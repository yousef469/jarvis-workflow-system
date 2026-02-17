import { useState, useCallback, useEffect } from 'react'
import FileExplorer from '../FileExplorer/FileExplorer'
import AIPanel from '../AIPanel/AIPanel'
import Terminal from '../Terminal/Terminal'
import { useFileStore } from '../../stores/fileStore'
import { useAIStore } from '../../stores/aiStore'

// Lazy-load Monaco
let MonacoEditor: any = null

export default function CodeMode() {
    const { openFiles, activeFilePath, setActiveFile, closeFile, updateFileContent, markFileSaved } = useFileStore()
    const { aiPanelOpen } = useAIStore()
    const [monacoLoaded, setMonacoLoaded] = useState(false)
    const [showTerminal, setShowTerminal] = useState(true)

    const activeFile = openFiles.find(f => f.path === activeFilePath)

    // Dynamic import Monaco
    useEffect(() => {
        import('@monaco-editor/react').then(mod => {
            MonacoEditor = mod.default
            setMonacoLoaded(true)
        })
    }, [])

    const getLanguage = (filePath: string): string => {
        const ext = filePath.split('.').pop()?.toLowerCase() || ''
        const langMap: Record<string, string> = {
            'js': 'javascript', 'jsx': 'javascript',
            'ts': 'typescript', 'tsx': 'typescript',
            'py': 'python', 'rs': 'rust',
            'c': 'c', 'cpp': 'cpp', 'h': 'cpp',
            'java': 'java', 'go': 'go',
            'html': 'html', 'css': 'css', 'scss': 'scss',
            'json': 'json', 'md': 'markdown',
            'yaml': 'yaml', 'yml': 'yaml',
            'sh': 'shell', 'bash': 'shell',
            'xml': 'xml', 'sql': 'sql',
        }
        return langMap[ext] || 'plaintext'
    }

    const handleSave = useCallback(async () => {
        if (!activeFile) return
        try {
            await window.jarvis.fs.writeFile(activeFile.path, activeFile.content)
            markFileSaved(activeFile.path)
        } catch (err) {
            console.error('Save failed:', err)
        }
    }, [activeFile, markFileSaved])

    // Keyboard shortcut for save
    useEffect(() => {
        const handler = (e: KeyboardEvent) => {
            if ((e.metaKey || e.ctrlKey) && e.key === 's') {
                e.preventDefault()
                handleSave()
            }
        }
        window.addEventListener('keydown', handler)
        return () => window.removeEventListener('keydown', handler)
    }, [handleSave])

    return (
        <div className="mode-code">
            <div className="code-layout">
                {/* File Explorer */}
                <FileExplorer />

                {/* Editor Area */}
                <div className="editor-area">
                    {/* Tab Bar */}
                    {openFiles.length > 0 && (
                        <div className="editor-tabs">
                            {openFiles.map(file => (
                                <div
                                    key={file.path}
                                    className={`editor-tab ${activeFilePath === file.path ? 'active' : ''}`}
                                    onClick={() => setActiveFile(file.path)}
                                >
                                    {file.isDirty && <span className="tab-dot" />}
                                    <span>{file.name}</span>
                                    <button
                                        className="tab-close"
                                        onClick={(e) => { e.stopPropagation(); closeFile(file.path) }}
                                    >
                                        ✕
                                    </button>
                                </div>
                            ))}
                        </div>
                    )}

                    {/* Editor */}
                    <div className="editor-wrapper">
                        {activeFile && monacoLoaded && MonacoEditor ? (
                            <MonacoEditor
                                height="100%"
                                language={getLanguage(activeFile.path)}
                                value={activeFile.content}
                                theme="vs-dark"
                                onChange={(value: string | undefined) => {
                                    if (value !== undefined) {
                                        updateFileContent(activeFile.path, value)
                                    }
                                }}
                                options={{
                                    fontSize: 13,
                                    fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
                                    minimap: { enabled: true },
                                    lineNumbers: 'on',
                                    renderWhitespace: 'selection',
                                    bracketPairColorization: { enabled: true },
                                    autoClosingBrackets: 'always',
                                    formatOnPaste: true,
                                    scrollBeyondLastLine: false,
                                    padding: { top: 8 },
                                    smoothScrolling: true,
                                    cursorBlinking: 'smooth',
                                    cursorSmoothCaretAnimation: 'on',
                                }}
                            />
                        ) : (
                            <div className="editor-empty">
                                <div className="editor-empty-icon">💻</div>
                                <p>Open a file to start editing</p>
                            </div>
                        )}
                    </div>

                    {/* Terminal */}
                    {showTerminal && <Terminal />}
                </div>

                {/* AI Panel */}
                {aiPanelOpen && <AIPanel />}
            </div>
        </div>
    )
}
