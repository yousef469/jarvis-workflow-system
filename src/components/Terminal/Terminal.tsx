import { useEffect, useRef, useCallback } from 'react'
import { useFileStore } from '../../stores/fileStore'

let Terminal: any = null
let FitAddon: any = null

export default function TerminalComponent() {
    const terminalRef = useRef<HTMLDivElement>(null)
    const termRef = useRef<any>(null)
    const fitAddonRef = useRef<any>(null)
    const termIdRef = useRef<string>('terminal-' + Math.random().toString(36).slice(2))
    const { workspacePath } = useFileStore()

    const initTerminal = useCallback(async () => {
        if (!terminalRef.current) return

        // Dynamic imports
        if (!Terminal) {
            const xtermMod = await import('@xterm/xterm')
            Terminal = xtermMod.Terminal

            // Import CSS manually
            const link = document.createElement('link')
            link.rel = 'stylesheet'
            link.href = 'https://cdn.jsdelivr.net/npm/@xterm/xterm@5.4.0/css/xterm.min.css'
            document.head.appendChild(link)
        }
        if (!FitAddon) {
            const fitMod = await import('@xterm/addon-fit')
            FitAddon = fitMod.FitAddon
        }

        const term = new Terminal({
            cursorBlink: true,
            cursorStyle: 'bar',
            fontSize: 12,
            fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
            theme: {
                background: '#0c0c14',
                foreground: '#e8e8f0',
                cursor: '#00f0ff',
                selectionBackground: 'rgba(0, 240, 255, 0.2)',
                black: '#0c0c14',
                red: '#ef4444',
                green: '#10b981',
                yellow: '#f59e0b',
                blue: '#0066ff',
                magenta: '#7c3aed',
                cyan: '#00f0ff',
                white: '#e8e8f0',
            },
            allowTransparency: true,
        })

        const fitAddon = new FitAddon()
        term.loadAddon(fitAddon)
        term.open(terminalRef.current)

        // Delay fit to ensure DOM is ready
        setTimeout(() => fitAddon.fit(), 100)

        termRef.current = term
        fitAddonRef.current = fitAddon

        // Create pty
        const cwd = workspacePath || process.env.HOME || '/'
        const id = termIdRef.current
        await window.jarvis.terminal.create(id, cwd)

        // Forward input to pty
        term.onData((data: string) => {
            window.jarvis.terminal.write(id, data)
        })

        // Receive output from pty
        const unsub = window.jarvis.terminal.onData((termId: string, data: string) => {
            if (termId === id) term.write(data)
        })

        // Handle resize
        const resizeObserver = new ResizeObserver(() => {
            fitAddon.fit()
            window.jarvis.terminal.resize(id, term.cols, term.rows)
        })
        if (terminalRef.current) resizeObserver.observe(terminalRef.current)

        return () => {
            unsub()
            resizeObserver.disconnect()
            term.dispose()
            window.jarvis.terminal.kill(id)
        }
    }, [workspacePath])

    useEffect(() => {
        const cleanup = initTerminal()
        return () => { cleanup?.then(fn => fn?.()) }
    }, [initTerminal])

    return (
        <div className="terminal-container">
            <div className="terminal-header">
                <span>⌨️ Terminal</span>
            </div>
            <div className="terminal-body" ref={terminalRef} />
        </div>
    )
}
