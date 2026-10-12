import React from 'react';
import { createRoot } from 'react-dom/client';
import { App } from './components/App';
import { BASENAME } from './host/env';
import './standalone.css';

const el = document.getElementById('root');
if (!el) {
  throw new Error('#root not found');
}
createRoot(el).render(<App basename={BASENAME} />);
