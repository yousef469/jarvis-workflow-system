import { useState, Suspense, useRef, useEffect, useCallback } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls, useGLTF, Html, Center } from '@react-three/drei'
import * as THREE from 'three'

declare global {
    interface Window {
        jarvis: any;
    }
}

// Simple spinning cube when no model is loaded
function SpinningCube() {
    const ref = useRef<THREE.Mesh>(null!)
    useFrame((_, delta) => {
        ref.current.rotation.x += delta * 0.3
        ref.current.rotation.y += delta * 0.5
    })
    return (
        <mesh ref={ref}>
            <boxGeometry args={[1.5, 1.5, 1.5]} />
            <meshPhongMaterial color="#6c63ff" shininess={80} />
        </mesh>
    )
}

// Ground grid for spatial reference
function Grid() {
    return (
        <gridHelper args={[20, 20, '#222', '#151525']} position={[0, -2, 0]} />
    )
}

// GLB model loader
function GLBModel({ url, onLoad }: { url: string; onLoad: () => void }) {
    const { scene } = useGLTF(url)
    useEffect(() => {
        if (scene) {
            // Auto-center and scale model
            const box = new THREE.Box3().setFromObject(scene)
            const size = box.getSize(new THREE.Vector3())
            const maxDim = Math.max(size.x, size.y, size.z)
            if (maxDim > 0) {
                const scale = 3 / maxDim
                scene.scale.setScalar(scale)
            }
            const center = box.getCenter(new THREE.Vector3())
            scene.position.sub(center.multiplyScalar(scene.scale.x))

            console.log('[3D] Model loaded:', scene.children.length, 'parts')
            onLoad()
        }
    }, [scene])
    return <primitive object={scene} />
}

export default function ThreeDMode() {
    const [blobUrl, setBlobUrl] = useState<string | null>(null)
    const [fileName, setFileName] = useState('')
    const [prompt, setPrompt] = useState('')
    const [generating, setGenerating] = useState(false)
    const [status, setStatus] = useState('')
    const [isDragging, setIsDragging] = useState(false)

    // Load file via jarvis-app protocol -> blob URL
    const loadFile = async (filePath: string) => {
        setStatus('Loading...')
        try {
            const res = await fetch(`jarvis-app://${filePath}`)
            if (!res.ok) throw new Error(`HTTP ${res.status}`)
            const blob = await res.blob()
            if (blobUrl) URL.revokeObjectURL(blobUrl)
            setBlobUrl(URL.createObjectURL(blob))
            setFileName(filePath.split('/').pop() || 'model.glb')
            setStatus('Rendering...')
        } catch (e: any) {
            setStatus(`Error: ${e.message}`)
        }
    }

    const handleOpen = async () => {
        const path = await window.jarvis.dialog.openFile()
        if (path) {
            const ext = path.split('.').pop()?.toLowerCase()
            if (['glb', 'gltf', 'stl'].includes(ext || '')) {
                loadFile(path)
            } else {
                setStatus('Select a .glb, .gltf, or .stl file')
            }
        }
    }

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault()
        setIsDragging(false)
        const file = e.dataTransfer.files[0] as any
        if (file?.path) {
            const ext = file.path.split('.').pop()?.toLowerCase()
            if (['glb', 'gltf', 'stl'].includes(ext || '')) loadFile(file.path)
        }
    }

    const handleGenerate = async () => {
        if (!prompt.trim()) return
        setGenerating(true)
        setStatus('Generating...')
        try {
            const res = await fetch('http://localhost:8765/api/3d/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt })
            })
            const data = await res.json()
            if (data.success && data.model_path) {
                await loadFile(data.model_path)
            } else {
                setStatus(`Failed: ${data.error || 'Unknown'}`)
            }
        } catch (e: any) {
            setStatus('Backend unreachable')
        } finally {
            setGenerating(false)
        }
    }

    const clearModel = () => {
        if (blobUrl) URL.revokeObjectURL(blobUrl)
        setBlobUrl(null)
        setFileName('')
        setStatus('')
    }

    return (
        <div
            onDragOver={e => { e.preventDefault(); setIsDragging(true) }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            style={{
                display: 'flex', flexDirection: 'column',
                height: '100%', width: '100%',
                background: '#08081a', color: '#e0e0e0',
                fontFamily: '-apple-system, BlinkMacSystemFont, sans-serif',
                position: 'relative'
            }}
        >
            {/* Header */}
            <header style={{
                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                padding: '14px 20px',
                borderBottom: '1px solid #1a1a2e',
                flexShrink: 0
            }}>
                <div>
                    <h2 style={{ margin: 0, fontSize: '17px', fontWeight: 600 }}>
                        🧊 3D Creator Studio
                    </h2>
                    {fileName && <p style={{ margin: '2px 0 0', fontSize: '11px', opacity: 0.4 }}>{fileName}</p>}
                </div>
                <div style={{ display: 'flex', gap: 8 }}>
                    <button onClick={handleOpen} style={btnS}>📂 Open</button>
                    {blobUrl && <button onClick={clearModel} style={{ ...btnS, color: '#ff6b6b' }}>✕ Clear</button>}
                </div>
            </header>

            {/* 3D Viewport - PERFORMANCE OPTIMIZED */}
            <div style={{ flex: 1, position: 'relative', minHeight: 0 }}>
                <Canvas
                    dpr={1}
                    camera={{ position: [4, 3, 5], fov: 45 }}
                    gl={{
                        antialias: false,
                        powerPreference: 'default',
                        alpha: false,
                        stencil: false,
                        depth: true
                    }}
                    style={{ width: '100%', height: '100%', display: 'block' }}
                >
                    <color attach="background" args={['#0a0a18']} />

                    {/* Minimal lighting for performance */}
                    <ambientLight intensity={0.8} />
                    <directionalLight position={[5, 8, 5]} intensity={1.5} />

                    <Grid />

                    {!blobUrl && <SpinningCube />}

                    {blobUrl && (
                        <Suspense fallback={
                            <Html center>
                                <div style={{ color: '#fff', background: '#1a1a2e', padding: '10px 20px', borderRadius: '10px', fontSize: '13px' }}>
                                    Loading mesh...
                                </div>
                            </Html>
                        }>
                            <GLBModel
                                url={blobUrl}
                                onLoad={() => setStatus('Model ready ✓')}
                            />
                        </Suspense>
                    )}

                    <OrbitControls
                        makeDefault
                        enableDamping={false}
                        rotateSpeed={0.8}
                        zoomSpeed={1.0}
                        panSpeed={0.8}
                    />
                </Canvas>

                {/* Status pill */}
                {status && (
                    <div style={{
                        position: 'absolute', top: 12, right: 12,
                        background: 'rgba(0,0,0,0.7)',
                        padding: '5px 12px', borderRadius: '16px',
                        fontSize: '11px', fontWeight: 500,
                        color: status.includes('Error') || status.includes('Failed') ? '#ff6b6b' : '#6c63ff',
                        border: '1px solid rgba(255,255,255,0.06)'
                    }}>
                        {status}
                    </div>
                )}

                {/* Drag overlay */}
                {isDragging && (
                    <div style={{
                        position: 'absolute', inset: 0,
                        background: 'rgba(108, 99, 255, 0.12)',
                        border: '2px dashed #6c63ff',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        zIndex: 50
                    }}>
                        <div style={{ textAlign: 'center' }}>
                            <div style={{ fontSize: 40 }}>📥</div>
                            <p style={{ margin: '8px 0 0', fontSize: 14 }}>Drop 3D Model</p>
                        </div>
                    </div>
                )}
            </div>

            {/* Generation Bar */}
            <div style={{
                padding: '14px 20px',
                borderTop: '1px solid #1a1a2e',
                display: 'flex', gap: 10,
                flexShrink: 0,
                background: '#0c0c1e'
            }}>
                <input
                    type="text"
                    value={prompt}
                    onChange={e => setPrompt(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && handleGenerate()}
                    disabled={generating}
                    placeholder="Describe a 3D model to generate..."
                    style={{
                        flex: 1, padding: '11px 16px',
                        background: '#08081a', color: '#fff',
                        border: '1px solid #1a1a2e',
                        borderRadius: '8px', fontSize: '14px',
                        outline: 'none'
                    }}
                />
                <button
                    onClick={handleGenerate}
                    disabled={generating || !prompt.trim()}
                    style={{
                        padding: '11px 22px', borderRadius: '8px',
                        background: generating ? '#333' : '#6c63ff',
                        color: '#fff', border: 'none',
                        cursor: generating ? 'wait' : 'pointer',
                        fontWeight: 600, fontSize: '14px',
                        opacity: (!prompt.trim() && !generating) ? 0.4 : 1
                    }}
                >
                    {generating ? '⏳ Creating...' : '✨ Generate'}
                </button>
            </div>
        </div>
    )
}

const btnS: React.CSSProperties = {
    padding: '6px 12px', background: 'transparent', color: '#ccc',
    border: '1px solid #1a1a2e', borderRadius: '6px', cursor: 'pointer', fontSize: '12px'
}
