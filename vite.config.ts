import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import electron from 'vite-plugin-electron/simple'
import renderer from 'vite-plugin-electron-renderer'
import { join } from 'path'

export default defineConfig({
    plugins: [
        react(),
        electron({
            main: {
                entry: 'electron/main/index.ts',
                vite: {
                    build: {
                        outDir: 'dist-electron/main',
                        rollupOptions: {
                            external: ['electron', 'node-pty', 'child_process', 'fs', 'path'],
                        },
                    },
                },
            },
            preload: {
                input: join(__dirname, 'electron/preload/index.ts'),
                vite: {
                    build: {
                        outDir: 'dist-electron/preload',
                    },
                },
            },
        }),
        renderer(),
    ],
    resolve: {
        alias: {
            '@': join(__dirname, 'src'),
        },
    },
    server: {
        host: '127.0.0.1',
        port: 5173,
    },
})
