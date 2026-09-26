import {
  LayoutDashboard,
  Users,
  BarChart3,
  Trash2,
  Cpu,
  ShieldCheck,
  GitCompare,
  Settings,
  Zap,
} from 'lucide-react';

export type Page =
  | 'dashboard'
  | 'clients'
  | 'contribution'
  | 'unlearning'
  | 'resource'
  | 'verification'
  | 'comparison'
  | 'settings';

const navItems: { id: Page; label: string; icon: React.ComponentType<{ size?: number; className?: string }> }[] = [
  { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { id: 'clients', label: 'Clients', icon: Users },
  { id: 'contribution', label: 'Contribution & Influence', icon: BarChart3 },
  { id: 'unlearning', label: 'Unlearning', icon: Trash2 },
  { id: 'resource', label: 'Resource Adaptation', icon: Cpu },
  { id: 'verification', label: 'Verification', icon: ShieldCheck },
  { id: 'comparison', label: 'Model Comparison', icon: GitCompare },
  { id: 'settings', label: 'Settings', icon: Settings },
];

interface SidebarProps {
  current: Page;
  onChange: (page: Page) => void;
}

export default function Sidebar({ current, onChange }: SidebarProps) {
  return (
    <aside className="fixed left-0 top-0 h-screen w-60 flex flex-col z-30" style={{ background: '#1a3352' }}>
      {/* Logo */}
      <div className="flex items-center gap-3 px-5 py-5 border-b" style={{ borderColor: 'rgba(255,255,255,0.08)' }}>
        <div
          className="w-9 h-9 rounded-xl flex items-center justify-center shrink-0"
          style={{ background: 'rgba(122,179,216,0.25)' }}
        >
          <Zap size={18} style={{ color: '#7ab3d8' }} />
        </div>
        <div>
          <div className="text-white font-display font-semibold text-base leading-tight tracking-tight">
            FedErase
          </div>
          <div className="text-xs leading-tight" style={{ color: 'rgba(255,255,255,0.45)', fontFamily: 'var(--font-mono)', letterSpacing: '0.04em' }}>
            Unlearning Framework
          </div>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 py-3 overflow-y-auto px-3 space-y-0.5">
        {navItems.map((item) => {
          const active = current === item.id;
          const Icon = item.icon;
          return (
            <button
              key={item.id}
              onClick={() => onChange(item.id)}
              className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium text-left cursor-pointer"
              style={{
                background: active ? 'rgba(122,179,216,0.18)' : 'transparent',
                color: active ? '#b8d8f0' : 'rgba(255,255,255,0.55)',
                borderLeft: active ? '3px solid #7ab3d8' : '3px solid transparent',
              }}
              onMouseEnter={(e) => {
                if (!active) {
                  (e.currentTarget as HTMLButtonElement).style.background = 'rgba(255,255,255,0.05)';
                  (e.currentTarget as HTMLButtonElement).style.color = 'rgba(255,255,255,0.8)';
                }
              }}
              onMouseLeave={(e) => {
                if (!active) {
                  (e.currentTarget as HTMLButtonElement).style.background = 'transparent';
                  (e.currentTarget as HTMLButtonElement).style.color = 'rgba(255,255,255,0.55)';
                }
              }}
            >
              <Icon size={16} />
              <span className="font-sans">{item.label}</span>
            </button>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="px-5 py-4 border-t" style={{ borderColor: 'rgba(255,255,255,0.08)' }}>
        <div className="text-xs" style={{ color: 'rgba(255,255,255,0.35)' }}>
          Federated Unlearning Framework
        </div>
        <div className="text-xs mt-0.5" style={{ color: 'rgba(255,255,255,0.25)', fontFamily: 'var(--font-mono)' }}>
          E16 • AI Research Project
        </div>
      </div>
    </aside>
  );
}
