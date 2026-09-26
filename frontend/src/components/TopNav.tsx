import { Bell, ChevronDown, Activity } from 'lucide-react';
import { useState } from 'react';

const pageTitles: Record<string, string> = {
  dashboard: 'Overview',
  clients: 'Client Management',
  contribution: 'Contribution & Influence',
  unlearning: 'Unlearning Pipeline',
  resource: 'Resource Adaptation',
  verification: 'Forgetting Verification',
  comparison: 'Model Comparison',
  settings: 'Settings',
};

interface TopNavProps {
  page: string;
  unreadNotifications?: number;
}

export default function TopNav({ page, unreadNotifications = 2 }: TopNavProps) {
  const [showNotif, setShowNotif] = useState(false);

  return (
    <header
      className="fixed top-0 right-0 h-14 flex items-center justify-between px-6 z-20"
      style={{
        left: '240px',
        background: 'rgba(245,243,238,0.9)',
        backdropFilter: 'blur(12px)',
        borderBottom: '1px solid var(--color-border)',
      }}
    >
      {/* Left: page title */}
      <div>
        <h1 className="font-display font-semibold text-base" style={{ color: '#1a3352' }}>
          {pageTitles[page] ?? page}
        </h1>
      </div>

      {/* Right: status + notifications + user */}
      <div className="flex items-center gap-3">
        {/* Model status */}
        <div
          className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium"
          style={{ background: 'rgba(45,106,159,0.08)', color: '#2d6a9f' }}
        >
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse-soft inline-block" />
          Model Active
        </div>

        {/* Round badge */}
        <div
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium"
          style={{ background: 'rgba(122,179,216,0.12)', color: '#1a3352' }}
        >
          <Activity size={12} />
          Round 20 / 20
        </div>

        {/* Notifications */}
        <div className="relative">
          <button
            onClick={() => setShowNotif((v) => !v)}
            className="relative w-8 h-8 rounded-lg flex items-center justify-center cursor-pointer"
            style={{ background: 'rgba(45,106,159,0.08)', color: '#6a8caa' }}
            onMouseEnter={(e) => ((e.currentTarget as HTMLButtonElement).style.background = 'rgba(45,106,159,0.14)')}
            onMouseLeave={(e) => ((e.currentTarget as HTMLButtonElement).style.background = 'rgba(45,106,159,0.08)')}
          >
            <Bell size={15} />
            {unreadNotifications > 0 && (
              <span
                className="absolute -top-0.5 -right-0.5 w-4 h-4 rounded-full flex items-center justify-center text-white text-[9px] font-bold"
                style={{ background: '#e05c5c' }}
              >
                {unreadNotifications}
              </span>
            )}
          </button>
          {showNotif && (
            <div
              className="absolute right-0 top-10 w-72 rounded-xl shadow-xl border overflow-hidden animate-fade-in"
              style={{ background: '#fff', borderColor: 'var(--color-border)' }}
            >
              <div className="px-4 py-3 border-b text-xs font-semibold uppercase tracking-wide" style={{ borderColor: 'var(--color-border)', color: '#6a8caa' }}>
                Notifications
              </div>
              {[
                { msg: 'Unlearning completed for C-04', sub: '2 minutes ago', dot: '#22c55e' },
                { msg: 'Client C-05 disconnected', sub: '2 hours ago', dot: '#f59e0b' },
              ].map((n, i) => (
                <div key={i} className="flex items-start gap-3 px-4 py-3 hover:bg-muted cursor-pointer" style={{ background: i === 0 ? 'rgba(45,106,159,0.04)' : 'transparent' }}>
                  <span className="mt-1.5 w-2 h-2 rounded-full shrink-0" style={{ background: n.dot }} />
                  <div>
                    <div className="text-sm font-medium" style={{ color: '#1a3352' }}>{n.msg}</div>
                    <div className="text-xs mt-0.5" style={{ color: '#6a8caa' }}>{n.sub}</div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* User */}
        <button
          className="flex items-center gap-2 px-2 py-1 rounded-lg cursor-pointer"
          style={{ color: '#1a3352' }}
          onMouseEnter={(e) => ((e.currentTarget as HTMLButtonElement).style.background = 'rgba(45,106,159,0.06)')}
          onMouseLeave={(e) => ((e.currentTarget as HTMLButtonElement).style.background = 'transparent')}
        >
          <div
            className="w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold text-white"
            style={{ background: '#2d6a9f' }}
          >
            E
          </div>
          <span className="text-sm font-medium hidden sm:block">Researcher</span>
          <ChevronDown size={12} className="hidden sm:block" style={{ color: '#6a8caa' }} />
        </button>
      </div>
    </header>
  );
}
