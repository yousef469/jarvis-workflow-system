import { useAppStore, type AppMode } from '../../stores/appStore'

const modes: { id: AppMode; icon: string; label: string }[] = [
    { id: 'chat', icon: '💬', label: 'AI Chat' },
    { id: '3d', icon: '🧊', label: '3D Viewer' },
    { id: 'sim', icon: '🔬', label: 'Simulations' },
    { id: 'image', icon: '🎨', label: 'Image Creation' },
    { id: 'video', icon: '🎬', label: 'Video Creation' },
    { id: '3dgen', icon: '🛠️', label: '3D Modeling' },
]

export default function Sidebar() {
    const { activeMode, setMode } = useAppStore()

    return (
        <div className="sidebar">
            <div className="sidebar-nav">
                {modes.map(mode => (
                    <button
                        key={mode.id}
                        className={`sidebar-btn ${activeMode === mode.id ? 'active' : ''}`}
                        onClick={() => setMode(mode.id)}
                    >
                        {mode.icon}
                        <span className="tooltip">{mode.label}</span>
                    </button>
                ))}
            </div>
            <div className="sidebar-bottom">
                <button
                    className={`sidebar-btn ${activeMode === 'settings' ? 'active' : ''}`}
                    onClick={() => setMode('settings')}
                >
                    ⚙️
                    <span className="tooltip">Settings</span>
                </button>
            </div>
        </div>
    )
}
