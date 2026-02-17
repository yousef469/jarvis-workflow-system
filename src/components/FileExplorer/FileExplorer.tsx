import { useState, useCallback } from 'react'
import { useFileStore, type FileEntry } from '../../stores/fileStore'

export default function FileExplorer() {
    const { workspacePath, fileTree, setWorkspacePath, setFileTree, toggleExpand, openFile } = useFileStore()

    const handleOpenFolder = useCallback(async () => {
        const folderPath = await window.jarvis.dialog.openFolder()
        if (!folderPath) return

        setWorkspacePath(folderPath)
        const entries = await loadDirectory(folderPath)
        setFileTree(entries)
    }, [setWorkspacePath, setFileTree])

    const loadDirectory = async (dirPath: string): Promise<FileEntry[]> => {
        try {
            const rawEntries = await window.jarvis.fs.readDirectory(dirPath)

            // Sort directories first, then files, alphabetically
            const sorted = rawEntries.sort((a, b) => {
                if (a.isDirectory !== b.isDirectory) return a.isDirectory ? -1 : 1
                return a.name.localeCompare(b.name)
            })

            // Filter out hidden/system files
            return sorted
                .filter(e => !e.name.startsWith('.') && e.name !== 'node_modules' && e.name !== '__pycache__' && e.name !== '.git')
                .map(e => ({
                    ...e,
                    isExpanded: false,
                    children: e.isDirectory ? [] : undefined
                }))
        } catch {
            return []
        }
    }

    const handleToggle = async (entry: FileEntry) => {
        if (!entry.isDirectory) return

        if (!entry.isExpanded && (!entry.children || entry.children.length === 0)) {
            const children = await loadDirectory(entry.path)
            // Update tree with children
            const updateTree = (entries: FileEntry[]): FileEntry[] =>
                entries.map(e => {
                    if (e.path === entry.path) return { ...e, children, isExpanded: true }
                    if (e.children) return { ...e, children: updateTree(e.children) }
                    return e
                })
            useFileStore.setState(s => ({ fileTree: updateTree(s.fileTree) }))
        } else {
            toggleExpand(entry.path)
        }
    }

    const handleFileClick = async (entry: FileEntry) => {
        if (entry.isDirectory) {
            handleToggle(entry)
            return
        }

        try {
            const content = await window.jarvis.fs.readFile(entry.path)
            openFile({
                path: entry.path,
                name: entry.name,
                content,
                isDirty: false
            })
        } catch (err) {
            console.error('Failed to open file:', err)
        }
    }

    const getFileIcon = (name: string, isDir: boolean): string => {
        if (isDir) return '📁'
        const ext = name.split('.').pop()?.toLowerCase() || ''
        const icons: Record<string, string> = {
            'py': '🐍', 'js': '📜', 'ts': '🔷', 'tsx': '⚛️', 'jsx': '⚛️',
            'html': '🌐', 'css': '🎨', 'scss': '🎨',
            'json': '📋', 'md': '📄', 'txt': '📄',
            'rs': '🦀', 'go': '🐹', 'cpp': '⚡', 'c': '⚡', 'h': '⚡',
            'java': '☕', 'rb': '💎',
            'png': '🖼️', 'jpg': '🖼️', 'svg': '🖼️',
            'pdf': '📕', 'docx': '📘',
            'stl': '🧊', 'glb': '🧊', 'gltf': '🧊',
        }
        return icons[ext] || '📄'
    }

    const renderTree = (entries: FileEntry[], depth = 0) => {
        return entries.map(entry => (
            <div key={entry.path}>
                <div
                    className={`file-tree-item ${useFileStore.getState().activeFilePath === entry.path ? 'active' : ''}`}
                    style={{ paddingLeft: `${8 + depth * 16}px` }}
                    onClick={() => handleFileClick(entry)}
                >
                    {entry.isDirectory && (
                        <span className={`chevron ${entry.isExpanded ? 'open' : ''}`}>▶</span>
                    )}
                    <span className="item-icon">{getFileIcon(entry.name, entry.isDirectory)}</span>
                    <span>{entry.name}</span>
                </div>
                {entry.isDirectory && entry.isExpanded && entry.children && (
                    <div className="file-tree-children">
                        {renderTree(entry.children, depth + 1)}
                    </div>
                )}
            </div>
        ))
    }

    return (
        <div className="file-explorer">
            <div className="file-explorer-header">
                <h3>Explorer</h3>
                <button onClick={handleOpenFolder} title="Open Folder">📂</button>
            </div>

            {!workspacePath ? (
                <div className="file-explorer-empty">
                    <p>No folder opened</p>
                    <button className="open-folder-btn" onClick={handleOpenFolder}>
                        📂 Open Folder
                    </button>
                </div>
            ) : (
                <div className="file-tree">
                    {renderTree(fileTree)}
                </div>
            )}
        </div>
    )
}
