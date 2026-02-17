import { useEffect } from 'react'
import { useAppStore } from './stores/appStore'
import Sidebar from './components/Sidebar/Sidebar.tsx'
import ChatMode from './components/Chat/ChatMode.tsx'
import ThreeDMode from './components/ThreeD/ThreeDMode.tsx'
import SimMode from './components/Simulations/SimMode.tsx'
import ImageMode from './components/ImageGen/ImageMode.tsx'
import { socketService } from './services/socket'

function App() {
    const { activeMode, backendStatus, setBackendStatus, addBackendLog, theme } = useAppStore()

    // Set initial theme
    useEffect(() => {
        document.documentElement.setAttribute('data-theme', theme)
    }, [theme])

    // Socket.IO Connection
    useEffect(() => {
        socketService.connect()
        return () => socketService.disconnect()
    }, [])

    // Check backend status periodically
    useEffect(() => {
        const check = async () => {
            try {
                const status = await window.jarvis.getBackendStatus()
                setBackendStatus(status as any)
            } catch {
                setBackendStatus('disconnected')
            }
        }

        check()
        const interval = setInterval(check, 10000)
        return () => clearInterval(interval)
    }, [setBackendStatus])

    // Listen for backend logs
    useEffect(() => {
        const unsub = window.jarvis.onBackendLog((msg: string) => {
            addBackendLog(msg)
        })
        return unsub
    }, [addBackendLog])

    const renderMode = () => {
        switch (activeMode) {
            case 'chat': return <ChatMode />
            case '3d': return <ThreeDMode />
            case 'sim': return <SimMode />
            case 'image': return <ImageMode />
            case 'video':
            case '3dgen':
            case 'settings': return (
                <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: 16 }}>
                    <div style={{ textAlign: 'center', color: 'var(--text-muted)' }}>
                        <div style={{ fontSize: 48, marginBottom: 12, opacity: 0.3 }}>
                            {activeMode === 'video' ? '🎬' : activeMode === '3dgen' ? '🛠️' : '⚙️'}
                        </div>
                        <h3 style={{ color: 'var(--text-secondary)', marginBottom: 8 }}>
                            {activeMode === 'video' ? 'Video Creation' : activeMode === '3dgen' ? '3D Modeling' : 'Settings'}
                        </h3>
                        <p style={{ fontSize: 13 }}>This feature is coming soon</p>
                    </div>
                </div>
            )
            default: return <ChatMode />
        }
    }

    return (
        <div className="app-container">
            {/* Titlebar */}
            <div className="titlebar">
                <div className="titlebar-left">
                    <div className="titlebar-brand">
                        <span className="brand-icon">◈</span>
                        <span>JARVIS</span>
                    </div>
                </div>
                <div className="titlebar-center">
                    {activeMode === 'chat' ? 'AI Assistant' :
                        activeMode === '3d' ? '3D Viewer' :
                            activeMode === 'sim' ? 'Simulations' :
                                activeMode === 'image' ? 'Image Creation' :
                                    activeMode === 'video' ? 'Video Creation' :
                                        activeMode === '3dgen' ? '3D Modeling' : 'Settings'}
                </div>
                <div className="titlebar-right">
                    <button onClick={useAppStore.getState().toggleTheme} title="Toggle theme">
                        {theme === 'dark' ? '☀️' : '🌙'}
                    </button>
                </div>
            </div>

            {/* Main Layout */}
            <div className="main-layout">
                <Sidebar />
                <div className="content-area">
                    {renderMode()}
                </div>
            </div>

            {/* Status Bar */}
            <div className="statusbar">
                <div className="statusbar-left">
                    <span>
                        <span className={`status-dot ${backendStatus}`}></span>
                        Jarvis {backendStatus === 'connected' ? 'Online' : backendStatus === 'loading' ? 'Connecting...' : 'Offline'}
                    </span>
                </div>
                <div className="statusbar-right">
                    <span>Mode: {activeMode}</span>
                    <span>Ollama</span>
                </div>
            </div>
        </div>
    )
}

export default App
