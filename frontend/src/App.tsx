import { useState, useEffect } from 'react';
import Sidebar, { type Page } from './components/Sidebar';
import TopNav from './components/TopNav';
import Dashboard from './pages/Dashboard';
import Clients from './pages/Clients';
import ContributionInfluence from './pages/ContributionInfluence';
import Unlearning from './pages/Unlearning';
import ResourceAdaptation from './pages/ResourceAdaptation';
import Verification from './pages/Verification';
import ModelComparison from './pages/ModelComparison';
import Settings from './pages/Settings';

interface Toast {
  id: number;
  msg: string;
  type: 'success' | 'info';
}

export default function App() {
  const [page, setPage] = useState<Page>('dashboard');
  const [targetClient, setTargetClient] = useState('');
  const [toasts, setToasts] = useState<Toast[]>([]);
  const nextId = useState(0)[0];
  let toastId = nextId;

  function toast(msg: string, type: 'success' | 'info' = 'info') {
    const id = ++toastId;
    setToasts((prev) => [...prev, { id, msg, type }]);
    setTimeout(() => setToasts((prev) => prev.filter((t) => t.id !== id)), 3500);
  }

  function navigate(p: Page) {
    setPage(p);
  }

  function startUnlearning() {
    navigate('unlearning');
    toast('Navigated to Unlearning Pipeline', 'info');
  }

  function selectForUnlearning(clientId: string) {
    setTargetClient(clientId);
    navigate('unlearning');
    toast(`Client ${clientId} selected for unlearning`, 'success');
  }

  // Page transition key for animation
  const [animKey, setAnimKey] = useState(0);
  useEffect(() => {
    setAnimKey((k) => k + 1);
  }, [page]);

  return (
    <div className="min-h-screen" style={{ background: 'var(--color-background)' }}>
      <Sidebar current={page} onChange={navigate} />

      {/* Main content area, offset by sidebar */}
      <div style={{ marginLeft: 240 }}>
        <TopNav page={page} />

        {/* Page content */}
        <main className="pt-14 px-6 pb-8 min-h-screen" style={{ maxWidth: 1400 }}>
          <div key={animKey} className="pt-6 animate-fade-in">
            {page === 'dashboard' && <Dashboard onStartUnlearning={startUnlearning} />}
            {page === 'clients' && <Clients onSelectForUnlearning={selectForUnlearning} />}
            {page === 'contribution' && <ContributionInfluence />}
            {page === 'unlearning' && <Unlearning initialClientId={targetClient} />}
            {page === 'resource' && <ResourceAdaptation />}
            {page === 'verification' && <Verification />}
            {page === 'comparison' && <ModelComparison />}
            {page === 'settings' && <Settings />}
          </div>
        </main>
      </div>

      {/* Toast notifications */}
      <div className="fixed bottom-6 right-6 z-50 flex flex-col gap-2 pointer-events-none">
        {toasts.map((t) => (
          <div
            key={t.id}
            className="toast px-4 py-3 rounded-xl shadow-lg flex items-center gap-2.5 text-sm font-medium"
            style={{
              background: t.type === 'success' ? '#f0fdf4' : '#f0f6fc',
              border: `1px solid ${t.type === 'success' ? '#b8e6c8' : '#d8e6f2'}`,
              color: t.type === 'success' ? '#166534' : '#1a3352',
              boxShadow: '0 4px 20px rgba(26,51,82,0.12)',
            }}
          >
            <span style={{ fontSize: 16 }}>{t.type === 'success' ? '✓' : 'ℹ'}</span>
            {t.msg}
          </div>
        ))}
      </div>
    </div>
  );
}
