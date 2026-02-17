import { create } from 'zustand'

export interface NoteSession {
    id: string
    title: string
    date: string
    notes: string[]
    keyTerms: string[]
    summary: string
    flashcards: { question: string; answer: string }[]
    examples?: string[]
    questions?: string[]
}

interface NotesStore {
    sessions: NoteSession[]
    activeSessionId: string | null
    isCapturing: boolean
    captureType: 'audio' | 'screen' | null

    addSession: (session: NoteSession) => void
    setActiveSession: (id: string | null) => void
    setIsCapturing: (v: boolean) => void
    setCaptureType: (type: 'audio' | 'screen' | null) => void
    deleteSession: (id: string) => void
}

export const useNotesStore = create<NotesStore>((set) => ({
    sessions: [],
    activeSessionId: null,
    isCapturing: false,
    captureType: null,

    addSession: (session) => set((state) => ({
        sessions: [session, ...state.sessions],
        activeSessionId: session.id
    })),
    setActiveSession: (id) => set({ activeSessionId: id }),
    setIsCapturing: (v) => set({ isCapturing: v }),
    setCaptureType: (type) => set({ captureType: type }),
    deleteSession: (id) => set((state) => ({
        sessions: state.sessions.filter(s => s.id !== id),
        activeSessionId: state.activeSessionId === id ? null : state.activeSessionId
    })),
}))
