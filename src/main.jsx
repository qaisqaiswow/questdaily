import React from 'react'
import ReactDOM from 'react-dom/client'
import SideQuestsApp from './SideQuestsApp'
import './index.css'
import { injectSpeedInsights } from '@vercel/speed-insights'

injectSpeedInsights()

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <SideQuestsApp />
  </React.StrictMode>,
)
