import { io, Socket } from 'socket.io-client'

class SocketService {
    private socket: Socket | null = null
    private url: string = 'http://localhost:8765' // Updated Jarvis backend port

    connect() {
        if (this.socket?.connected) return

        this.socket = io(this.url, {
            reconnectionAttempts: 5,
            reconnectionDelay: 1000,
            autoConnect: true
        })

        this.socket.on('connect', () => {
            console.log('Connected to Jarvis backend via Socket.IO')
        })

        this.socket.on('disconnect', () => {
            console.log('Disconnected from Jarvis backend')
        })

        this.socket.on('error', (err) => {
            console.error('Socket.IO Error:', err)
        })
    }

    on(event: string, callback: (...args: any[]) => void) {
        this.socket?.on(event, callback)
    }

    emit(event: string, data: any) {
        this.socket?.emit(event, data)
    }

    disconnect() {
        this.socket?.disconnect()
        this.socket = null
    }

    isConnected() {
        return this.socket?.connected || false
    }
}

export const socketService = new SocketService()
export default socketService
