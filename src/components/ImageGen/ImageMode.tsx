import { useState, useEffect } from 'react'
import '../../styles/image-mode.css'

interface ModelInfo {
    image: string
    brain: string
}

export default function ImageMode() {
    const [prompt, setPrompt] = useState('')
    const [generating, setGenerating] = useState(false)
    const [currentImage, setCurrentImage] = useState<string | null>(null)
    const [recentImages, setRecentImages] = useState<string[]>([])
    const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null)
    const [status, setStatus] = useState('Ready')

    // Fetch model info on mount
    useEffect(() => {
        fetch('http://localhost:8765/api/system/models')
            .then(res => res.json())
            .then(data => setModelInfo(data))
            .catch(err => console.error("Failed to fetch models:", err))
    }, [])

    const handleGenerate = async () => {
        if (!prompt.trim() || generating) return

        setGenerating(true)
        setStatus('Initializing Image Pipeline...')

        try {
            // 1. Send generation request
            setStatus('Generating 1024×1024 image (~8 min)...')
            const res = await fetch('http://localhost:8765/api/image/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt })
            })

            const data = await res.json()

            if (data.success && data.path) {
                // Convert to jarvis-app protocol for local file access
                const imageUrl = `jarvis-app://${data.path}`
                console.log(`[ImageMode] Success! Resolved image path: ${imageUrl}`)

                setCurrentImage(imageUrl)
                setRecentImages(prev => [imageUrl, ...prev].slice(0, 10))
                setStatus(`Generated: "${data.prompt || prompt}"`)
            } else {
                console.error("[ImageMode] Generation failed:", data)
                setStatus(`Error: ${data.error || 'Generation failed'}`)
            }
        } catch (e: any) {
            console.error("[ImageMode] Connection Error:", e)
            setStatus(`Connection Error: ${e.message}`)
        } finally {
            setGenerating(false)
        }
    }

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            handleGenerate()
        }
    }

    return (
        <div className="image-mode-container">
            {/* Header */}
            <header className="image-header">
                <h2>
                    <span>🎨</span> Image Creation Studio
                </h2>
                {modelInfo && (
                    <div className="model-info-pill">
                        Model: {modelInfo.image}
                    </div>
                )}
            </header>

            {/* Main Preview Area */}
            <div className="image-preview-area">
                {currentImage ? (
                    <img src={currentImage} alt="Generated result" className="image-result" />
                ) : (
                    <div className="empty-state">
                        <div style={{ fontSize: 48, marginBottom: 16 }}>🖼️</div>
                        <p>Enter a prompt below to start creating.</p>
                        <p style={{ fontSize: 12 }}>Powered by SDXL Lightning • ~8 min per image</p>
                    </div>
                )}

                {/* Status Overlay */}
                {status && (
                    <div style={{
                        position: 'absolute', bottom: 20,
                        background: 'rgba(0,0,0,0.6)', padding: '6px 14px',
                        borderRadius: 20, fontSize: 12,
                        backdropFilter: 'blur(4px)',
                        color: status.includes('Error') ? '#ff6b6b' : '#fff'
                    }}>
                        {status}
                    </div>
                )}
            </div>

            {/* Controls */}
            <div className="image-controls">
                {/* Recent Gallery (Small Thumbs) */}
                {recentImages.length > 0 && (
                    <div className="recent-gallery">
                        {recentImages.map((img, idx) => (
                            <div
                                key={idx}
                                className={`recent-thumb ${currentImage === img ? 'active' : ''}`}
                                onClick={() => setCurrentImage(img)}
                            >
                                <img src={img} alt={`Recent ${idx}`} />
                            </div>
                        ))}
                    </div>
                )}

                {/* Input Area */}
                <div className="prompt-input-group">
                    <input
                        className="prompt-input"
                        placeholder="Describe what you want to see... (e.g. 'A futuristic city with neon lights')"
                        value={prompt}
                        onChange={(e) => setPrompt(e.target.value)}
                        onKeyDown={handleKeyDown}
                        disabled={generating}
                        autoFocus
                    />
                    <button
                        className="generate-btn"
                        onClick={handleGenerate}
                        disabled={generating || !prompt.trim()}
                    >
                        {generating ? (
                            <><span>⏳</span> Creating...</>
                        ) : (
                            <><span>✨</span> Generate</>
                        )}
                    </button>
                </div>
            </div>
        </div>
    )
}
