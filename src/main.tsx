import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './styles/variables.css'
import './styles/global.css'
import './styles/layout.css'
import './styles/sidebar.css'
import './styles/chat.css'
import './styles/editor.css'
import './styles/file-explorer.css'
import './styles/ai-panel.css'
import './styles/notes.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
    <React.StrictMode>
        <App />
    </React.StrictMode>
)
