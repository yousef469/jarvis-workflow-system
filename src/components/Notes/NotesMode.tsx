import { useState, useRef, useEffect } from 'react'
import { useNotesStore, type NoteSession } from '../../stores/notesStore'

export default function NotesMode() {
    const {
        sessions,
        activeSessionId,
        setActiveSession,
        addSession,
        isCapturing,
        setIsCapturing,
        setCaptureType
    } = useNotesStore()

    const [textInput, setTextInput] = useState('')
    const [processing, setProcessing] = useState(false)
    const [status, setStatus] = useState('Ready')
    const [liveNotes, setLiveNotes] = useState<any[]>([])
    const [generating, setGenerating] = useState(false)

    // Media Refs
    const audioInterval = useRef<any>(null)
    const mediaRecorder = useRef<MediaRecorder | null>(null)
    const [hasActiveStream, setHasActiveStream] = useState(false)
    const screenStream = useRef<MediaStream | null>(null)
    const captureInterval = useRef<any>(null)
    const videoRef = useRef<HTMLVideoElement | null>(null)
    const canvasRef = useRef<HTMLCanvasElement | null>(null)
    const liveFeedRef = useRef<HTMLDivElement | null>(null)

    const transcriptRef = useRef('')
    const visualNotesRef = useRef('')

    // Active session display
    const activeSession = sessions.find(s => s.id === activeSessionId)

    // Reliability: Re-attach stream whenever video element mounts
    useEffect(() => {
        if (isCapturing && hasActiveStream && screenStream.current && videoRef.current) {
            console.log('[Notetaker] Attaching stream to video element')
            const video = videoRef.current
            video.srcObject = screenStream.current
            video.play().catch(err => console.warn('[Notetaker] Video play failed:', err))
        }
    }, [isCapturing, hasActiveStream])

    // Auto-scroll live feed
    useEffect(() => {
        if (liveFeedRef.current) {
            liveFeedRef.current.scrollTop = liveFeedRef.current.scrollHeight
        }
    }, [liveNotes])

    // --- Audio/Screen Capture Logic ---

    const [sources, setSources] = useState<any[]>([])
    const [showPicker, setShowPicker] = useState(false)

    const startCapture = async (type: 'audio' | 'screen') => {
        try {
            setStatus('Fetching sources...')
            const availableSources = await (window as any).jarvis.getSources()
            setSources(availableSources)
            setShowPicker(true)
            setStatus('Select a source below')
        } catch (err) {
            console.error('Failed to get sources:', err)
            setStatus('Error: Failed to fetch sources')
        }
    }

    const selectSource = async (sourceId: string) => {
        try {
            setShowPicker(false)
            setStatus('Connecting to source...')
            setLiveNotes([])
            transcriptRef.current = ''
            visualNotesRef.current = ''

            // 1. Capture Video using Source ID
            const stream = await (navigator.mediaDevices as any).getUserMedia({
                audio: false,
                video: {
                    mandatory: {
                        chromeMediaSource: 'desktop',
                        chromeMediaSourceId: sourceId,
                        minWidth: 1280,
                        maxWidth: 1920,
                        minHeight: 720,
                        maxHeight: 1080,
                        minFrameRate: 30,
                        maxFrameRate: 60
                    }
                }
            })

            screenStream.current = stream
            setHasActiveStream(true)
            setIsCapturing(true)
            setCaptureType('screen')

            // 2. Add Audio Stream (Microphone) for Whisper
            try {
                const audioStream = await navigator.mediaDevices.getUserMedia({ audio: true })
                const audioTrack = audioStream.getAudioTracks()[0]
                console.log('[Notetaker] Audio source selected:', audioTrack.label)
                setupAudioRecording(audioStream)
                setStatus('🔴 Live (Video + Audio)')
            } catch (aErr) {
                console.warn('Microphone access denied, audio disabled:', aErr)
                setStatus('🔴 Live (Video ONLY)')
            }

            // 3. Start Analysis Intervals
            captureInterval.current = setInterval(captureFrame, 5000)
            captureFrame()

            setIsCapturing(true)
            setCaptureType('screen')

            stream.getVideoTracks()[0].onended = stopCapture
        } catch (error: any) {
            setStatus('Error: ' + error.message)
            console.error('Source selection failed:', error)
        }
    }

    const setupAudioRecording = (stream: MediaStream) => {
        const createRecorder = () => {
            const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm' })
            recorder.ondataavailable = async (e) => {
                if (e.data.size > 0) {
                    const blob = new Blob([e.data], { type: 'audio/webm' })
                    const reader = new FileReader()
                    reader.onloadend = async () => {
                        const base64 = (reader.result as string).split(',')[1]
                        try {
                            const res = await fetch('http://localhost:8765/api/live/audio', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ audioData: base64, mimeType: 'audio/webm' })
                            })
                            const data = await res.json()
                            console.log('[Notetaker] Audio Response:', data)
                            if (data.notes && data.notes.length > 0) {
                                data.notes.forEach((note: any) => {
                                    console.log('[Notetaker] Adding transcript:', note.text)
                                    transcriptRef.current += ' ' + note.text
                                    addLiveNote('📝', note.text, 'transcript')
                                })
                            }
                        } catch (err) {
                            console.error('Audio chunk error:', err)
                        }
                    }
                    reader.readAsDataURL(blob)
                }
            }
            return recorder
        }

        mediaRecorder.current = createRecorder()
        mediaRecorder.current.start()

        // Cycle recorder every 1 second for true live STT
        audioInterval.current = setInterval(() => {
            if (mediaRecorder.current && mediaRecorder.current.state === 'recording') {
                mediaRecorder.current.stop()
                mediaRecorder.current = createRecorder()
                mediaRecorder.current.start()
            }
        }, 1000)
    }

    const [emptyCaptureCount, setEmptyCaptureCount] = useState(0)
    const [lastCapturedImage, setLastCapturedImage] = useState<string | null>(null)

    const captureFrame = async () => {
        let imageData: string | null = null
        let fullDataUrl: string | null = null

        // Try standard browser capture first if stream is active
        if (screenStream.current && canvasRef.current && videoRef.current) {
            const video = videoRef.current
            const canvas = canvasRef.current
            if (video.videoWidth > 0) {
                const ctx = canvas.getContext('2d')
                if (ctx) {
                    canvas.width = video.videoWidth
                    canvas.height = video.videoHeight
                    ctx.drawImage(video, 0, 0)
                    fullDataUrl = canvas.toDataURL('image/jpeg', 0.6)
                    imageData = fullDataUrl.split(',')[1]
                }
            }
        }

        // Fallback: Direct IPC Capture from Main Process
        if (!imageData) {
            console.log('[Notetaker] Browser capture unavailable, trying direct IPC capture...')
            fullDataUrl = await (window as any).jarvis.captureFrame()
            if (fullDataUrl) {
                // If it's just a tiny placeholder or empty, dataUrl will be small
                if (fullDataUrl.length < 1000) {
                    console.warn('[Notetaker] IPC Capture returned suspicious/empty data.')
                    fullDataUrl = null
                } else {
                    imageData = fullDataUrl.split(',')[1]
                }
            }
        }

        if (fullDataUrl) {
            setLastCapturedImage(fullDataUrl)
        }

        if (!imageData) {
            console.warn('[Notetaker] Capture frame failed.')
            setEmptyCaptureCount(prev => prev + 1)
            if (emptyCaptureCount > 3) {
                setStatus('⚠️ Permission Denied (macOS Screen Recording)')
            }
            return
        }

        setEmptyCaptureCount(0)
        console.log('[Notetaker] Sending frame for analysis...')
        try {
            const res = await fetch('http://localhost:8765/api/live/frame', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ imageData, context: visualNotesRef.current.slice(-500) })
            })
            const data = await res.json()
            if (data.notes && data.notes.length > 0) {
                console.log(`[Notetaker] Received ${data.notes.length} visual notes`)
                data.notes.forEach((note: any) => {
                    visualNotesRef.current += '\n' + note.text
                    // Use 👁️ icon for OCR text
                    addLiveNote('👁️', note.text, 'ocr')
                })
            } else {
                console.log('[Notetaker] No new visual content detected')
            }
        } catch (err) {
            console.error('Frame analysis error:', err)
        }
    }

    const addLiveNote = (icon: string, text: string, type = 'small') => {
        const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        setLiveNotes(prev => [...prev, { time, icon, text, type, id: Date.now() }])
    }

    const stopCapture = async () => {
        setStatus('Generating Final Notes...')
        setGenerating(true)

        if (captureInterval.current) clearInterval(captureInterval.current)
        if (audioInterval.current) clearInterval(audioInterval.current)
        if (mediaRecorder.current && mediaRecorder.current.state === 'recording') mediaRecorder.current.stop()
        if (screenStream.current) {
            screenStream.current.getTracks().forEach(t => t.stop())
            screenStream.current = null
        }
        setHasActiveStream(false)
        setLastCapturedImage(null)

        setIsCapturing(false)
        setCaptureType(null)

        try {
            const res = await fetch('http://localhost:8765/api/live/finalize', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ transcript: transcriptRef.current, visualNotes: visualNotesRef.current })
            })
            const data = await res.json()
            const final = data.notes

            if (final) {
                const session: NoteSession = {
                    id: Date.now().toString(),
                    title: final.title || 'Lecture Notes',
                    date: new Date().toLocaleString(),
                    notes: final.bullet_points || [],
                    keyTerms: (final.key_terms || []).map((t: any) => t.term),
                    summary: final.summary || '',
                    flashcards: (final.flashcards || []).map((f: any) => ({ question: f.q, answer: f.a })),
                    examples: final.examples || [],
                    questions: final.questions || []
                }
                addSession(session)
                setStatus('✅ Done')
            }
        } catch (err: any) {
            setStatus('Error: ' + err.message)
        } finally {
            setGenerating(false)
        }
    }

    const handlePasteGenerate = async () => {
        if (!textInput.trim()) return
        setProcessing(true)
        setStatus('Processing text...')

        try {
            const res = await fetch('http://localhost:8765/api/notes/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: textInput })
            })
            const data = await res.json()
            const final = data.notes

            if (final) {
                const session: NoteSession = {
                    id: Date.now().toString(),
                    title: final.topic || 'Pasted Notes',
                    date: new Date().toLocaleString(),
                    notes: final.bullet_points || [],
                    keyTerms: (final.key_terms || []).map((t: any) => t.term),
                    summary: final.summary || '',
                    flashcards: (final.flashcards || []).map((f: any) => ({ question: f.q, answer: f.a })),
                    examples: [],
                    questions: final.questions || []
                }
                addSession(session)
                setTextInput('')
                setStatus('✅ Done')
            }
        } catch (err) {
            console.error('Generation failed:', err)
            setStatus('❌ Failed')
        } finally {
            setProcessing(false)
        }
    }

    return (
        <div className="notes-container">
            {/* Left: Notes List */}
            <div className="notes-sidebar">
                <div className="notes-sidebar-header">
                    <h3>📝 Notes</h3>
                    <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{sessions.length} sessions</span>
                </div>
                <div className="notes-list">
                    {sessions.length === 0 ? (
                        <div style={{ padding: 16, textAlign: 'center', color: 'var(--text-muted)', fontSize: 12 }}>
                            No sessions yet.
                        </div>
                    ) : (
                        sessions.map(session => (
                            <div
                                key={session.id}
                                className={`note-item ${activeSessionId === session.id ? 'active' : ''}`}
                                onClick={() => setActiveSession(session.id)}
                            >
                                <div className="note-item-title">{session.title}</div>
                                <div className="note-item-date">{session.date}</div>
                            </div>
                        ))
                    )}
                </div>
            </div>

            {/* Right: Notes Content */}
            <div className="notes-main">
                <div className="notes-toolbar">
                    <div className="status-badge">
                        <span className={`status-dot ${isCapturing ? 'active' : ''}`}></span>
                        {status}
                    </div>

                    {!isCapturing && !generating && (
                        <>
                            <button className="notes-toolbar-btn primary" onClick={() => startCapture('screen')}>
                                🚀 Start Live Session
                            </button>
                            <div style={{ flex: 1 }} />
                            {textInput && (
                                <button className="notes-toolbar-btn" onClick={handlePasteGenerate} disabled={processing}>
                                    {processing ? '...' : '✨ Process Paste'}
                                </button>
                            )}
                        </>
                    )}

                    {(isCapturing || generating) && (
                        <button className="notes-toolbar-btn stop" onClick={stopCapture} disabled={generating}>
                            ⏹ Stop & Finalize
                        </button>
                    )}
                </div>

                <div className="notes-content">
                    {showPicker && (
                        <div className="source-picker-overlay" onClick={(e) => e.stopPropagation()}>
                            <div className="source-picker-header">
                                <h3>🎥 Select Video Source</h3>
                                <button className="source-picker-close" onClick={() => setShowPicker(false)}>
                                    Cancel
                                </button>
                            </div>
                            <div className="source-picker-grid">
                                {sources.length === 0 ? (
                                    <div style={{ padding: 20, textAlign: 'center', gridColumn: '1/-1', opacity: 0.7 }}>
                                        <p>No windows or screens detected.</p>
                                        <button
                                            className="notes-toolbar-btn"
                                            style={{ marginTop: 10 }}
                                            onClick={() => startCapture('screen')}
                                        >
                                            🔄 Retry Fetching Sources
                                        </button>
                                    </div>
                                ) : (
                                    sources.map(source => (
                                        <div
                                            key={source.id}
                                            className="source-item"
                                            onClick={() => selectSource(source.id)}
                                        >
                                            <img src={source.thumbnail} className="source-thumbnail" alt={source.name} />
                                            <div className="source-name">{source.name}</div>
                                        </div>
                                    ))
                                )}
                            </div>
                        </div>
                    )}

                    {isCapturing ? (
                        <div className="live-capture-view">
                            <div className="video-section" style={{ position: 'relative' }}>
                                <div style={{
                                    position: 'absolute',
                                    top: 10,
                                    left: 10,
                                    zIndex: 10,
                                    background: 'rgba(0,0,0,0.6)',
                                    padding: '4px 8px',
                                    fontSize: 10,
                                    borderRadius: 4,
                                    color: (hasActiveStream && videoRef.current && videoRef.current.videoWidth > 0) ? '#00f0ff' : '#ff4444',
                                    pointerEvents: 'none'
                                }}>
                                    {isCapturing && (hasActiveStream && videoRef.current && videoRef.current.videoWidth > 0)
                                        ? `LIVE: ${videoRef.current.videoWidth}x${videoRef.current.videoHeight}`
                                        : 'PREVIEW (Awaiting Live Stream...)'
                                    }
                                </div>

                                <video
                                    ref={videoRef}
                                    autoPlay
                                    muted
                                    playsInline
                                    style={{
                                        display: (hasActiveStream && videoRef.current && videoRef.current.videoWidth > 0) ? 'block' : 'none',
                                        width: '100%',
                                        height: '100%'
                                    }}
                                />
                                {(lastCapturedImage && (!hasActiveStream || !videoRef.current || videoRef.current.videoWidth === 0)) && (
                                    <img
                                        src={lastCapturedImage}
                                        alt="Live Preview"
                                        style={{
                                            width: '100%',
                                            height: 'auto',
                                            borderRadius: 8,
                                            border: '1px solid var(--accent-primary)',
                                            display: 'block',
                                            boxShadow: '0 0 20px rgba(0, 240, 255, 0.2)'
                                        }}
                                    />
                                )}
                                {(!hasActiveStream && !lastCapturedImage) && (
                                    <div style={{ padding: 40, textAlign: 'center', opacity: 0.5 }}>
                                        <p style={{ margin: 0, fontSize: 13 }}>Initializing Multi-Modal Pipeline...</p>
                                        <small style={{ fontSize: 10 }}>Checking macOS permissions...</small>
                                    </div>
                                )}
                                <canvas ref={canvasRef} style={{ display: 'none' }} />
                            </div>
                            <div className="live-feed" ref={liveFeedRef}>
                                <h4>LIVE TRANSCRIPTION</h4>
                                {liveNotes.map(n => (
                                    <div key={n.id} className={`live-note ${n.type}`}>
                                        <span className="icon">{n.icon}</span>
                                        <div className="text">
                                            <small>{n.time}</small>
                                            <p>{n.text}</p>
                                        </div>
                                    </div>
                                ))}
                                {liveNotes.length === 0 && <p style={{ opacity: 0.5, fontSize: 12 }}>Detecting content...</p>}
                            </div>
                        </div>
                    ) : activeSession ? (
                        <div className="notes-display">
                            <h2 className="notes-title">{activeSession.title}</h2>

                            <div className="notes-grid">
                                <div className="notes-col-main">
                                    <div className="notes-section">
                                        <h3>📋 Summary</h3>
                                        <div className="note-card summary">
                                            <p>{activeSession.summary}</p>
                                        </div>
                                    </div>

                                    <div className="notes-section">
                                        <h3>📝 Detailed Notes</h3>
                                        <div className="points-list">
                                            {activeSession.notes.map((p, i) => (
                                                <div key={i} className="point-item">{p}</div>
                                            ))}
                                        </div>
                                    </div>

                                    {activeSession.examples && activeSession.examples.length > 0 && (
                                        <div className="notes-section">
                                            <h3>💡 Examples & Analogies</h3>
                                            {activeSession.examples.map((ex, i) => (
                                                <div key={i} className="note-card example">{ex}</div>
                                            ))}
                                        </div>
                                    )}
                                </div>

                                <div className="notes-col-side">
                                    <div className="notes-section">
                                        <h3>🔑 Key Terms</h3>
                                        <div className="terms-cloud">
                                            {activeSession.keyTerms.map((t, i) => <span key={i} className="term-chip">{t}</span>)}
                                        </div>
                                    </div>

                                    {activeSession.flashcards.length > 0 && (
                                        <div className="notes-section">
                                            <h3>🃏 Flashcards</h3>
                                            {activeSession.flashcards.map((fc, i) => (
                                                <FlashCard key={i} question={fc.question} answer={fc.answer} />
                                            ))}
                                        </div>
                                    )}

                                    {activeSession.questions && activeSession.questions.length > 0 && (
                                        <div className="notes-section">
                                            <h3>❓ Review Questions</h3>
                                            <div className="questions-list">
                                                {activeSession.questions.map((q, i) => <div key={i} className="question-item">{q}</div>)}
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>
                    ) : (
                        <div className="notes-empty-state">
                            <div className="empty-hero">
                                <span className="hero-icon">🔬</span>
                                <h2>Jarvis Intelligence Research</h2>
                                <p>Record a lecture, share a video, or paste text to generate professional-grade documentation.</p>
                            </div>
                            <textarea
                                value={textInput}
                                onChange={e => setTextInput(e.target.value)}
                                placeholder="Paste research data, transcripts, or notes here for instant AI structuring..."
                            />
                        </div>
                    )}
                </div>
            </div>
        </div>
    )
}

function FlashCard({ question, answer }: { question: string; answer: string }) {
    const [revealed, setRevealed] = useState(false)
    return (
        <div className="flashcard-compact" onClick={() => setRevealed(!revealed)}>
            <div className="fc-q">Q: {question}</div>
            {revealed ? <div className="fc-a">A: {answer}</div> : <div className="fc-hint">Click to reveal</div>}
        </div>
    )
}
