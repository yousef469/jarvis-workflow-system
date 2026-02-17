import { create } from 'zustand'

export interface ChatMessage {
    role: 'user' | 'assistant' | 'system'
    content: string
    timestamp: number
    image?: string
}

export type AiCodeMode = 'analyze' | 'explain' | 'fix' | 'generate' | 'chat'

interface AIStore {
    // Chat mode
    chatMessages: ChatMessage[]
    chatLoading: boolean

    // Code AI mode
    aiCodeMode: AiCodeMode
    codeAnalysis: any | null
    aiPanelOpen: boolean

    // Voice
    isListening: boolean
    isRecording: boolean

    // Actions
    addChatMessage: (msg: ChatMessage) => void
    updateLastChatMessage: (content: string) => void
    setChatLoading: (v: boolean) => void
    clearChat: () => void
    setAiCodeMode: (mode: AiCodeMode) => void
    setCodeAnalysis: (analysis: any) => void
    toggleAiPanel: () => void
    setIsListening: (v: boolean) => void
    setIsRecording: (v: boolean) => void
}

export const useAIStore = create<AIStore>((set) => ({
    chatMessages: [],
    chatLoading: false,
    aiCodeMode: 'analyze',
    codeAnalysis: null,
    aiPanelOpen: true,
    isListening: false,
    isRecording: false,

    addChatMessage: (msg) => set((state) => ({
        chatMessages: [...state.chatMessages, msg]
    })),
    updateLastChatMessage: (content) => set((state) => {
        const last = state.chatMessages[state.chatMessages.length - 1]
        if (!last || last.role !== 'assistant') return state
        const updated = { ...last, content: last.content + content }
        return {
            chatMessages: [...state.chatMessages.slice(0, -1), updated]
        }
    }),
    setChatLoading: (v) => set({ chatLoading: v }),
    clearChat: () => set({ chatMessages: [] }),
    setAiCodeMode: (mode) => set({ aiCodeMode: mode }),
    setCodeAnalysis: (analysis) => set({ codeAnalysis: analysis }),
    toggleAiPanel: () => set((state) => ({ aiPanelOpen: !state.aiPanelOpen })),
    setIsListening: (v) => set({ isListening: v }),
    setIsRecording: (v) => set({ isRecording: v }),
}))
