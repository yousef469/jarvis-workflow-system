import { useState, useEffect } from 'react'

export default function SimMode() {
    const [status, setStatus] = useState('Idle')
    const [progress, setProgress] = useState(0)
    const [logs, setLogs] = useState<string[]>([])

    const addLog = (msg: string) => {
        setLogs(prev => [`[${new Date().toLocaleTimeString()}] ${msg}`, ...prev.slice(0, 19)])
    }

    const startSim = () => {
        setStatus('Running')
        setProgress(0)
        addLog('Initializing structural analysis engine...')
    }

    useEffect(() => {
        if (status === 'Running') {
            const timer = setInterval(() => {
                setProgress(prev => {
                    if (prev >= 100) {
                        clearInterval(timer)
                        setStatus('Completed')
                        addLog('Simulation finished. Results saved to /exports/sim_01.json')
                        return 100
                    }
                    if (prev === 20) addLog('Mesh generation complete (42,000 nodes)')
                    if (prev === 50) addLog('Solving Navier-Stokes equations...')
                    if (prev === 80) addLog('Converging on steady-state solution...')
                    return prev + 1
                })
            }, 100)
            return () => clearInterval(timer)
        }
    }, [status])

    return (
        <div className="sim-mode-container">
            <div className="sim-header">
                <div className="sim-info">
                    <h3>🔬 Physics & Simulation</h3>
                    <p>Finite Element Analysis (FEA) & Computational Fluid Dynamics (CFD)</p>
                </div>
                <div className="sim-controls">
                    <button
                        onClick={startSim}
                        className="primary-btn"
                        disabled={status === 'Running'}
                    >
                        {status === 'Running' ? '⟳ Running...' : '▶ Start Simulation'}
                    </button>
                    <button
                        onClick={() => { setStatus('Idle'); setProgress(0); setLogs([]) }}
                        className="secondary-btn"
                    >
                        ✕ Reset
                    </button>
                </div>
            </div>

            <div className="sim-grid">
                {/* Visualizer Placeholder */}
                <div className="sim-card visualizer">
                    <div className="card-header">
                        <span>Live Visualizer</span>
                        <div className="badge blue">GPU Accelerated</div>
                    </div>
                    <div className="sim-viz-content">
                        {status === 'Running' || status === 'Completed' ? (
                            <div className="viz-placeholder active">
                                <div className="viz-mesh"></div>
                                <div className="viz-overlay">
                                    <div className="viz-stat">
                                        <span>Velocity:</span>
                                        <strong>{(progress * 1.2).toFixed(1)} m/s</strong>
                                    </div>
                                    <div className="viz-stat">
                                        <span>Pressure:</span>
                                        <strong>{(progress * 0.5 + 101.3).toFixed(1)} kPa</strong>
                                    </div>
                                </div>
                            </div>
                        ) : (
                            <div className="viz-placeholder idle">
                                <div style={{ fontSize: 48, opacity: 0.1, marginBottom: 12 }}>≋</div>
                                <p>Ready for analysis</p>
                            </div>
                        )}
                    </div>
                </div>

                {/* Status Column */}
                <div className="sim-card-stack">
                    <div className="sim-card status">
                        <div className="card-header">
                            <span>System Status</span>
                            <div className={`badge ${status === 'Running' ? 'orange' : status === 'Completed' ? 'green' : 'gray'}`}>
                                {status.toUpperCase()}
                            </div>
                        </div>
                        <div className="sim-progress-area">
                            <div className="progress-label">
                                <span>Total Progress</span>
                                <span>{progress}%</span>
                            </div>
                            <div className="progress-bar-bg">
                                <div className="progress-bar-fill" style={{ width: `${progress}%` }}></div>
                            </div>
                        </div>
                    </div>

                    <div className="sim-card logs">
                        <div className="card-header">
                            <span>Real-time Logs</span>
                        </div>
                        <div className="sim-log-list">
                            {logs.map((log, i) => (
                                <div key={i} className="sim-log-item">{log}</div>
                            ))}
                            {logs.length === 0 && <div className="empty-logs">No activity</div>}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}
