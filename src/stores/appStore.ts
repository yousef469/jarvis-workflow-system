import { create } from 'zustand'

export type AppMode = 'chat' | 'notes' | '3d' | 'sim' | 'video' | 'image' | '3dgen' | 'settings'

interface AppState {
    activeMode: AppMode
    backendStatus: 'disconnected' | 'connected' | 'loading'
    theme: 'dark' | 'light'
    sidebarExpanded: boolean
    backendLogs: string[]

    setMode: (mode: AppMode) => void
    setBackendStatus: (status: 'disconnected' | 'connected' | 'loading') => void
    toggleTheme: () => void
    toggleSidebar: () => void
    addBackendLog: (log: string) => void
}

export const useAppStore = create<AppState>((set) => ({
    activeMode: 'chat',
    backendStatus: 'disconnected',
    theme: 'dark',
    sidebarExpanded: false,
    backendLogs: [],

    setMode: (mode) => set({ activeMode: mode }),
    setBackendStatus: (status) => set({ backendStatus: status }),
    toggleTheme: () => set((state) => {
        const next = state.theme === 'dark' ? 'light' : 'dark'
        document.documentElement.setAttribute('data-theme', next)
        return { theme: next }
    }),
    toggleSidebar: () => set((state) => ({ sidebarExpanded: !state.sidebarExpanded })),
    addBackendLog: (log) => set((state) => ({
        backendLogs: [...state.backendLogs.slice(-100), log]
    })),
}))
